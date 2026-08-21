// OpenAI-compatible HTTP server.
//
// Single sequence, serialised by a mutex: the directive puts continuous batching
// explicitly out of scope, so a request holds the engine for its duration and
// others queue. `/metrics` exposes the queue depth so that is visible rather
// than mysterious.
//
// Three endpoints exist because local-agent-bootstrap needs them and would
// silently break without them, not because OpenAI defines them:
//   * GET /health          -- every lifecycle command polls it
//   * a `timings` object    -- `agent bench` hard-exits without it
//   * a configurable model id matching /v1/models[0].id
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include "../../third_party/httplib.h"
#include "../../third_party/json.hpp"
#include "../model/model.h"
#include "../tokenizer/bpe.h"
#include "../tokenizer/chat_template.h"
#include "../kernels/sampling.cuh"
#include "../kernels/elementwise.cuh"
#include "reasoning_parser.h"
#include "server_config.h"

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

namespace qwen {

struct Metrics {
  std::atomic<uint64_t> requests{0}, errors{0};
  std::atomic<uint64_t> prompt_tokens{0}, completion_tokens{0}, reasoning_tokens{0};
  std::atomic<uint64_t> cached_prefix_tokens{0};
  std::atomic<uint64_t> verify_steps{0}, drafted{0}, accepted{0};
  std::atomic<uint64_t> queue_depth{0};
  std::atomic<double>   last_decode_tok_s{0.0}, last_prefill_tok_s{0.0}, last_ttft_s{0.0};
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};

namespace {

Model*        g_model = nullptr;
Tokenizer*    g_tok = nullptr;
SamplerState  g_sampler;
ServerConfig  g_cfg;
Metrics       g_metrics;
std::mutex    g_engine;
uint64_t      g_step_counter = 0;

std::string now_id(const char* prefix) {
  static std::atomic<uint64_t> n{0};
  char b[64];
  snprintf(b, sizeof b, "%s-%llu", prefix, (unsigned long long)(++n));
  return b;
}

int64_t unix_now() {
  return std::chrono::duration_cast<std::chrono::seconds>(
             std::chrono::system_clock::now().time_since_epoch()).count();
}

json error_json(const std::string& msg, const char* type, int code) {
  return json{{"error", {{"message", msg}, {"type", type}, {"code", code}}}};
}

// ---------------------------------------------------------------- params
SamplingParams parse_sampling(const json& b, bool thinking) {
  SamplingParams p;
  // Directive 8.6 defaults, which differ by mode.
  if (thinking) { p.temperature = 1.0f; p.top_p = 0.95f; p.top_k = 20; p.presence_penalty = 0.0f; }
  else          { p.temperature = 0.7f; p.top_p = 0.80f; p.top_k = 20; p.presence_penalty = 1.5f; }
  if (b.contains("temperature") && !b["temperature"].is_null()) p.temperature = b["temperature"].get<float>();
  if (b.contains("top_p") && !b["top_p"].is_null())             p.top_p = b["top_p"].get<float>();
  if (b.contains("top_k") && !b["top_k"].is_null())             p.top_k = b["top_k"].get<int>();
  if (b.contains("min_p") && !b["min_p"].is_null())             p.min_p = b["min_p"].get<float>();
  if (b.contains("presence_penalty") && !b["presence_penalty"].is_null())
    p.presence_penalty = b["presence_penalty"].get<float>();
  if (b.contains("frequency_penalty") && !b["frequency_penalty"].is_null())
    p.frequency_penalty = b["frequency_penalty"].get<float>();
  if (b.contains("repetition_penalty") && !b["repetition_penalty"].is_null())
    p.repetition_penalty = b["repetition_penalty"].get<float>();
  if (b.contains("seed") && !b["seed"].is_null()) p.seed = b["seed"].get<uint64_t>();
  return p;
}

struct GenResult {
  std::string content, reasoning;
  std::vector<ToolCall> calls;
  int prompt_tokens = 0, completion_tokens = 0, reasoning_tokens = 0;
  double ttft_s = 0, decode_tok_s = 0, prefill_tok_s = 0;
  std::string finish_reason = "stop";
};

// The single generation path. `on_delta` is called with each decoded piece; a
// non-streaming request just accumulates.
GenResult generate(const std::vector<int32_t>& ids, const SamplingParams& sp,
                   int max_new, const std::vector<std::string>& stops,
                   bool starts_in_think,
                   const std::function<bool(const std::string&, const std::string&)>& on_delta) {
  GenResult r;
  Model& m = *g_model;
  const auto t0 = std::chrono::steady_clock::now();

  sampler_reset_counts(g_sampler);
  cudaMemset(m.gdn_state, 0, size_t(m.shape.gdn_state_elems()) * 4);
  cudaMemset(m.gdn_conv, 0, size_t(m.shape.gdn_conv_state_elems()) * 4);

  // chunked prefill
  int pos = 0;
  const int chunk = m.max_batch;
  while (pos < int(ids.size())) {
    const int n = std::min<int>(chunk, int(ids.size()) - pos);
    model_prefill(m, ids.data() + pos, n, pos);
    pos += n;
  }
  cudaDeviceSynchronize();
  const auto t_prefill = std::chrono::steady_clock::now();
  r.prompt_tokens = int(ids.size());
  r.prefill_tok_s = ids.size() /
      std::chrono::duration<double>(t_prefill - t0).count();

  // The chat template's generation prompt ends with "<think>\n" when thinking is
  // enabled, so the model starts INSIDE the reasoning block and the splitter
  // never sees an opening tag -- it would put the whole chain of thought into
  // `content` and leak a bare "</think>".
  ReasoningSplitter sp_split(starts_in_think);
  std::string all;
  int32_t* d_id = m.argmax_scratch + 512;
  bool first = true;
  for (int i = 0; i < max_new; ++i) {
    sample(d_id, m.logits, g_sampler, sp, g_step_counter++);
    int32_t tok = 0;
    cudaMemcpy(&tok, d_id, 4, cudaMemcpyDeviceToHost);
    if (first) {
      r.ttft_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      first = false;
    }
    if (m.shape.is_stop_token(tok)) { r.finish_reason = "stop"; break; }
    sampler_note_token(g_sampler, tok);
    ++r.completion_tokens;

    const std::string piece = g_tok->decode({tok});
    all += piece;
    std::string c, th;
    sp_split.feed(piece, false, c, th);
    if (sp_split.in_think() || !th.empty()) r.reasoning_tokens += 1;
    r.content += c; r.reasoning += th;
    if ((!c.empty() || !th.empty()) && on_delta && !on_delta(c, th)) {
      r.finish_reason = "abort"; break;
    }

    bool stopped = false;
    for (const auto& s : stops)
      if (!s.empty() && r.content.size() >= s.size() &&
          r.content.compare(r.content.size() - s.size(), s.size(), s) == 0) {
        r.content.erase(r.content.size() - s.size());
        stopped = true; break;
      }
    if (stopped) { r.finish_reason = "stop"; break; }
    if (i + 1 == max_new) r.finish_reason = "length";
    model_decode(m, tok, pos + i);
  }
  std::string c, th;
  sp_split.feed("", true, c, th);
  r.content += c; r.reasoning += th;
  if (on_delta && (!c.empty() || !th.empty())) on_delta(c, th);
  r.calls = sp_split.tool_calls();

  const auto t1 = std::chrono::steady_clock::now();
  const double dec = std::chrono::duration<double>(t1 - t_prefill).count();
  r.decode_tok_s = dec > 0 ? r.completion_tokens / dec : 0;
  return r;
}

json usage_json(const GenResult& r, int cached) {
  return json{{"prompt_tokens", r.prompt_tokens},
              {"completion_tokens", r.completion_tokens},
              {"total_tokens", r.prompt_tokens + r.completion_tokens},
              {"prompt_tokens_details", {{"cached_tokens", cached}}},
              {"completion_tokens_details", {{"reasoning_tokens", r.reasoning_tokens}}}};
}

// llama.cpp-shaped timings. local-agent-bootstrap's `agent bench` hard-exits
// without this object, taking `agent code` and `agent status` with it.
json timings_json(const GenResult& r) {
  return json{{"prompt_n", r.prompt_tokens},
              {"prompt_per_second", r.prefill_tok_s},
              {"predicted_n", r.completion_tokens},
              {"predicted_per_second", r.decode_tok_s}};
}

json tool_calls_json(const std::vector<ToolCall>& calls) {
  json arr = json::array();
  for (size_t i = 0; i < calls.size(); ++i)
    arr.push_back({{"id", "call_" + std::to_string(i)},
                   {"type", "function"},
                   {"index", int(i)},
                   {"function", {{"name", calls[i].name},
                                 {"arguments", calls[i].arguments_json}}}});
  return arr;
}

}  // namespace

int run_server(Model& model, Tokenizer& tok, const ServerConfig& cfg) {
  g_model = &model; g_tok = &tok; g_cfg = cfg;
  sampler_alloc(g_sampler, model.shape.vocab_size);

  httplib::Server srv;
  srv.set_payload_max_length(64ull << 20);

  auto cors = [](httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Headers", "*");
    res.set_header("Access-Control-Allow-Methods", "*");
  };
  srv.Options(".*", [&](const httplib::Request&, httplib::Response& res) {
    cors(res); res.status = 204;
  });

  // llama.cpp endpoint, not OpenAI. Body is never read; only the status matters.
  srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    cors(res);
    res.set_content(json{{"status", "ok"},
                         {"model", g_cfg.model_alias},
                         {"max_context", g_cfg.max_ctx},
                         {"queue", int(g_metrics.queue_depth.load())}}.dump(),
                    "application/json");
  });

  srv.Get("/", [&](const httplib::Request&, httplib::Response& res) {
    cors(res);
    FILE* fp = fopen((g_cfg.webui_path).c_str(), "rb");
    if (!fp) {
      res.set_content("qwen38-3090-cuda-server\nendpoints: /health /v1/models "
                      "/v1/chat/completions /v1/completions /metrics\n", "text/plain");
      return;
    }
    std::string body;
    char b[8192]; size_t n;
    while ((n = fread(b, 1, sizeof b, fp)) > 0) body.append(b, n);
    fclose(fp);
    res.set_content(body, "text/html; charset=utf-8");
  });

  srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
    cors(res);
    res.set_content(json{{"object", "list"},
                         {"data", json::array({json{{"id", g_cfg.model_alias},
                                                    {"object", "model"},
                                                    {"created", unix_now()},
                                                    {"owned_by", "local"}}})}}.dump(),
                    "application/json");
  });

  srv.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
    cors(res);
    const double up = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - g_metrics.start).count();
    const uint64_t vs = g_metrics.verify_steps.load(), ac = g_metrics.accepted.load();
    std::string o;
    auto add = [&](const char* name, const char* help, const char* type, double v) {
      o += std::string("# HELP ") + name + " " + help + "\n";
      o += std::string("# TYPE ") + name + " " + type + "\n";
      char b[128]; snprintf(b, sizeof b, "%s %.6g\n", name, v); o += b;
    };
    add("qwen_uptime_seconds", "Server uptime.", "counter", up);
    add("qwen_requests_total", "Requests served.", "counter", double(g_metrics.requests));
    add("qwen_errors_total", "Requests that returned an error.", "counter", double(g_metrics.errors));
    add("qwen_prompt_tokens_total", "Prompt tokens processed.", "counter", double(g_metrics.prompt_tokens));
    add("qwen_completion_tokens_total", "Tokens generated.", "counter", double(g_metrics.completion_tokens));
    add("qwen_reasoning_tokens_total", "Tokens inside <think>.", "counter", double(g_metrics.reasoning_tokens));
    add("qwen_cached_prefix_tokens_total", "Prompt tokens served from the prefix cache.", "counter", double(g_metrics.cached_prefix_tokens));
    add("qwen_queue_depth", "Requests waiting for the single engine slot.", "gauge", double(g_metrics.queue_depth));
    add("qwen_decode_tokens_per_second", "Decode rate of the last request.", "gauge", g_metrics.last_decode_tok_s);
    add("qwen_prefill_tokens_per_second", "Prefill rate of the last request.", "gauge", g_metrics.last_prefill_tok_s);
    add("qwen_ttft_seconds", "Time to first token of the last request.", "gauge", g_metrics.last_ttft_s);
    add("qwen_spec_verify_steps_total", "Speculative verification steps.", "counter", double(vs));
    add("qwen_spec_drafted_tokens_total", "Tokens proposed by the drafter.", "counter", double(g_metrics.drafted));
    add("qwen_spec_accepted_tokens_total", "Drafted tokens the target accepted.", "counter", double(ac));
    // Mean acceptance length counts the always-free committed token, so 1.0
    // means speculation bought nothing.
    add("qwen_spec_mean_acceptance_length", "Committed tokens per verification step.",
        "gauge", vs ? double(ac + vs) / double(vs) : 0.0);
    res.set_content(o, "text/plain; version=0.0.4");
  });

  auto handle = [&](const httplib::Request& req, httplib::Response& res, bool chat) {
    cors(res);
    g_metrics.requests++;
    json b;
    try { b = json::parse(req.body); }
    catch (const std::exception& e) {
      g_metrics.errors++;
      res.status = 400;
      res.set_content(error_json(std::string("invalid JSON: ") + e.what(),
                                 "invalid_request_error", 400).dump(), "application/json");
      return;
    }

    // ---- build the prompt ----
    std::vector<int32_t> ids;
    bool thinking = true;
    try {
      if (chat) {
        if (!b.contains("messages") || !b["messages"].is_array())
          throw ChatTemplateError("`messages` is required and must be an array");
        ChatOptions opt;
        opt.add_generation_prompt = true;
        if (b.contains("enable_thinking") && !b["enable_thinking"].is_null())
          opt.enable_thinking = b["enable_thinking"].get<bool>();
        if (b.contains("preserve_thinking") && !b["preserve_thinking"].is_null())
          opt.preserve_thinking = b["preserve_thinking"].get<bool>();
        if (b.contains("reasoning_effort") && b["reasoning_effort"].is_string())
          opt.reasoning_effort = b["reasoning_effort"].get<std::string>();
        thinking = !opt.enable_thinking.has_value() || *opt.enable_thinking;

        ojson msgs = ojson::parse(b["messages"].dump());
        // OpenAI clients send tool_call arguments as a JSON STRING; the chat
        // template iterates `arguments|items` and raises on a string. Parse it.
        for (auto& msg : msgs) {
          if (!msg.contains("tool_calls") || !msg["tool_calls"].is_array()) continue;
          for (auto& tc : msg["tool_calls"]) {
            if (!tc.contains("function")) continue;
            auto& fn = tc["function"];
            if (fn.contains("arguments") && fn["arguments"].is_string()) {
              try { fn["arguments"] = ojson::parse(fn["arguments"].get<std::string>()); }
              catch (...) { fn["arguments"] = ojson::object(); }
            }
          }
        }
        ojson tools = b.contains("tools") ? ojson::parse(b["tools"].dump()) : ojson();
        const std::string prompt = render_chat(msgs, tools, opt);
        ids = g_tok->encode(prompt);
      } else {
        if (!b.contains("prompt") || !b["prompt"].is_string())
          throw ChatTemplateError("`prompt` is required and must be a string");
        ids = g_tok->encode(b["prompt"].get<std::string>());
      }
    } catch (const UnsupportedContent& e) {
      g_metrics.errors++;
      res.status = 400;
      res.set_content(error_json(e.what(), "invalid_request_error", 400).dump(),
                      "application/json");
      return;
    } catch (const std::exception& e) {
      g_metrics.errors++;
      res.status = 400;
      res.set_content(error_json(e.what(), "invalid_request_error", 400).dump(),
                      "application/json");
      return;
    }

    if (int(ids.size()) >= g_cfg.max_ctx) {
      g_metrics.errors++;
      res.status = 400;
      res.set_content(error_json("prompt of " + std::to_string(ids.size()) +
                                 " tokens exceeds --max-context " +
                                 std::to_string(g_cfg.max_ctx),
                                 "invalid_request_error", 400).dump(), "application/json");
      return;
    }

    int max_new = 512;
    for (const char* k : {"max_tokens", "max_completion_tokens", "n_predict"})
      if (b.contains(k) && b[k].is_number()) { max_new = b[k].get<int>(); break; }
    max_new = std::min(max_new, g_cfg.max_ctx - int(ids.size()) - 1);

    std::vector<std::string> stops;
    if (b.contains("stop")) {
      if (b["stop"].is_string()) stops.push_back(b["stop"].get<std::string>());
      else if (b["stop"].is_array()) for (auto& s : b["stop"]) stops.push_back(s.get<std::string>());
    }
    const SamplingParams sp = parse_sampling(b, thinking);
    const bool stream = b.contains("stream") && b["stream"].is_boolean() && b["stream"].get<bool>();
    bool want_usage = true;
    if (b.contains("stream_options") && b["stream_options"].is_object() &&
        b["stream_options"].contains("include_usage"))
      want_usage = b["stream_options"]["include_usage"].get<bool>();
    else if (stream) want_usage = false;   // OpenAI default: omit unless asked

    const std::string id = now_id(chat ? "chatcmpl" : "cmpl");
    const int64_t created = unix_now();

    if (!stream) {
      g_metrics.queue_depth++;
      std::lock_guard<std::mutex> lk(g_engine);
      g_metrics.queue_depth--;
      GenResult r = generate(ids, sp, max_new, stops, thinking, nullptr);
      g_metrics.prompt_tokens += r.prompt_tokens;
      g_metrics.completion_tokens += r.completion_tokens;
      g_metrics.reasoning_tokens += r.reasoning_tokens;
      g_metrics.last_decode_tok_s = r.decode_tok_s;
      g_metrics.last_prefill_tok_s = r.prefill_tok_s;
      g_metrics.last_ttft_s = r.ttft_s;

      json out;
      if (chat) {
        json msg{{"role", "assistant"}, {"content", r.content}};
        if (!r.reasoning.empty()) msg["reasoning_content"] = r.reasoning;
        if (!r.calls.empty()) {
          msg["tool_calls"] = tool_calls_json(r.calls);
          r.finish_reason = "tool_calls";
        }
        out = json{{"id", id}, {"object", "chat.completion"}, {"created", created},
                   {"model", g_cfg.model_alias},
                   {"choices", json::array({json{{"index", 0}, {"message", msg},
                                                 {"finish_reason", r.finish_reason}}})}};
      } else {
        out = json{{"id", id}, {"object", "text_completion"}, {"created", created},
                   {"model", g_cfg.model_alias},
                   {"choices", json::array({json{{"index", 0}, {"text", r.content},
                                                 {"finish_reason", r.finish_reason}}})}};
      }
      out["usage"] = usage_json(r, 0);
      out["timings"] = timings_json(r);
      res.set_content(out.dump(), "application/json");
      return;
    }

    // ---- streaming ----
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");
    res.set_header("X-Accel-Buffering", "no");
    res.set_chunked_content_provider(
        "text/event-stream",
        [id, created, ids, sp, max_new, stops, chat, want_usage, thinking]
        (size_t, httplib::DataSink& sink) {
          auto emit = [&](const json& j) {
            const std::string s = "data: " + j.dump() + "\n\n";
            return sink.write(s.data(), s.size());
          };
          auto chunk = [&](json delta, const json& finish) {
            return json{{"id", id}, {"object", chat ? "chat.completion.chunk" : "text_completion"},
                        {"created", created}, {"model", g_cfg.model_alias},
                        {"choices", json::array({json{{"index", 0}, {"delta", delta},
                                                      {"finish_reason", finish}}})}};
          };
          g_metrics.queue_depth++;
          std::lock_guard<std::mutex> lk(g_engine);
          g_metrics.queue_depth--;
          // Role prelude first: it puts a byte on the wire immediately, which
          // matters because clients set a TTFB deadline (OpenCode uses 120 s)
          // and a long prefill would otherwise blow it.
          if (chat) emit(chunk(json{{"role", "assistant"}}, nullptr));

          bool alive = true;
          GenResult r = generate(ids, sp, max_new, stops, thinking,
              [&](const std::string& c, const std::string& th) {
                json d = json::object();
                if (!c.empty())  d["content"] = c;
                if (!th.empty()) d["reasoning_content"] = th;
                if (d.empty()) return true;
                alive = emit(chunk(d, nullptr));
                return alive;
              });
          g_metrics.prompt_tokens += r.prompt_tokens;
          g_metrics.completion_tokens += r.completion_tokens;
          g_metrics.reasoning_tokens += r.reasoning_tokens;
          g_metrics.last_decode_tok_s = r.decode_tok_s;
          g_metrics.last_prefill_tok_s = r.prefill_tok_s;
          g_metrics.last_ttft_s = r.ttft_s;

          if (alive && !r.calls.empty()) {
            emit(chunk(json{{"tool_calls", tool_calls_json(r.calls)}}, nullptr));
            r.finish_reason = "tool_calls";
          }
          if (alive) emit(chunk(json::object(), r.finish_reason));
          // OpenAI emits usage as its OWN chunk with an empty choices array,
          // and only when stream_options.include_usage asked for it.
          if (alive && want_usage) {
            json u{{"id", id}, {"object", "chat.completion.chunk"}, {"created", created},
                   {"model", g_cfg.model_alias}, {"choices", json::array()},
                   {"usage", usage_json(r, 0)}, {"timings", timings_json(r)}};
            emit(u);
          }
          if (alive) { const std::string d = "data: [DONE]\n\n"; sink.write(d.data(), d.size()); }
          sink.done();
          return true;
        });
  };

  srv.Post("/v1/chat/completions", [&](const httplib::Request& q, httplib::Response& r) { handle(q, r, true); });
  srv.Post("/v1/completions", [&](const httplib::Request& q, httplib::Response& r) { handle(q, r, false); });
  srv.Post("/completion", [&](const httplib::Request& q, httplib::Response& r) { handle(q, r, false); });

  printf("listening on http://%s:%d  (model id: %s, max_context %d)\n",
         cfg.host.c_str(), cfg.port, cfg.model_alias.c_str(), cfg.max_ctx);
  fflush(stdout);
  if (!srv.listen(cfg.host.c_str(), cfg.port)) {
    fprintf(stderr, "failed to bind %s:%d\n", cfg.host.c_str(), cfg.port);
    return 1;
  }
  return 0;
}

}  // namespace qwen
