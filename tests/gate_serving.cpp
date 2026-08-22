// Production-readiness gate for the HTTP server, run against a LIVE instance.
//
// Every other gate in this repo tests a kernel or a load path in isolation.
// This one tests the thing a coding agent actually talks to: the OpenAI-shaped
// endpoint, with the prefix cache, the drafter and the INT4 KV cache all live at
// once. Three of the four bugs found while making the GGUF path work were
// invisible to every kernel gate and would have been caught here in seconds --
// a full prefix hit aborting the server, --max-context auto serving a context of
// 0, and cached_tokens reporting 0 on every request.
//
// It is deliberately weight-format agnostic. `tools/run_serving_gates.sh` starts
// the server once per configuration (AWQ and GGUF, both with INT4 KV and the
// drafter) and runs this against each, so "production ready" means the same
// assertions pass on both runners rather than on whichever one was developed
// last.
//
// EVERY request carries a timeout and every prompt is bounded, so a hung server
// fails this gate rather than hanging it.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

static int g_pass = 0, g_fail = 0;
static std::string g_host = "127.0.0.1";
static int g_port = 8099;

static void ok(const char* name, const std::string& detail = "") {
  printf("  [PASS] %-42s %s\n", name, detail.c_str());
  ++g_pass;
}
static void bad(const char* name, const std::string& why) {
  printf("  [FAIL] %-42s %s\n", name, why.c_str());
  ++g_fail;
}
static void check(const char* name, bool cond, const std::string& detail) {
  if (cond) ok(name, detail); else bad(name, detail);
}

static httplib::Client mk(int timeout_s) {
  httplib::Client c(g_host, g_port);
  c.set_connection_timeout(5, 0);
  c.set_read_timeout(timeout_s, 0);
  c.set_write_timeout(30, 0);
  return c;
}

// POST a chat completion and parse it. Returns an empty json on transport error.
static json post(const json& body, int timeout_s = 180, const char* path = "/v1/chat/completions",
                 int* status = nullptr) {
  auto c = mk(timeout_s);
  auto r = c.Post(path, body.dump(), "application/json");
  if (!r) { if (status) *status = -1; return json(); }
  if (status) *status = r->status;
  try { return json::parse(r->body); } catch (...) { return json(); }
}

static json user_msg(const std::string& text) {
  return json::array({json{{"role", "user"}, {"content", text}}});
}

// A greedy request. Thinking is disabled where the template allows it, so the
// assertions are about `content` rather than about how long the model reasons.
static json greedy(const json& messages, int max_tokens, bool stream = false) {
  json b{{"model", "qwen38-27b"}, {"messages", messages}, {"temperature", 0},
         {"max_tokens", max_tokens}, {"chat_template_kwargs", {{"enable_thinking", false}}}};
  if (stream) { b["stream"] = true; b["stream_options"] = {{"include_usage", true}}; }
  return b;
}

static std::string content_of(const json& r) {
  if (!r.contains("choices") || r["choices"].empty()) return "";
  const json& m = r["choices"][0]["message"];
  std::string s = m.value("content", "");
  if (s.empty()) s = m.value("reasoning_content", "");
  return s;
}

static double metric(const std::string& body, const std::string& name) {
  const size_t p = body.find("\n" + name + " ");
  if (p == std::string::npos) return -1;
  return atof(body.c_str() + p + name.size() + 2);
}

static std::string get_metrics() {
  auto c = mk(20);
  auto r = c.Get("/metrics");
  return r ? r->body : std::string();
}

// A prompt of roughly `n_words` words with a needle near the front, so a long
// prompt is a RETRIEVAL test rather than just a memory-allocation test.
static std::string long_prompt(int n_words, const std::string& needle) {
  std::string s = "You are reading a project log. " + needle + "\n\n";
  static const char* w[] = {"module", "commit", "refactor", "buffer", "kernel", "latency",
                            "queue", "token", "cache", "stride", "layer", "tensor"};
  for (int i = 0; i < n_words; ++i) {
    s += w[i % 12];
    s += (i % 17 == 16) ? ".\n" : " ";
  }
  return s;
}

// ---------------------------------------------------------------- checks

static void check_liveness() {
  auto c = mk(20);
  auto h = c.Get("/health");
  check("health 200", h && h->status == 200, h ? std::to_string(h->status) : "no response");
  auto m = c.Get("/v1/models");
  bool shaped = false;
  if (m && m->status == 200) {
    try {
      json j = json::parse(m->body);
      shaped = j.contains("data") && !j["data"].empty() && j["data"][0].contains("id");
    } catch (...) {}
  }
  check("/v1/models is OpenAI-shaped", shaped, "");
}

static void check_greedy_determinism() {
  // What IS a contract: the same request down the same path is bit-identical.
  // Calls 2 and 3 both hit the prefix cache in full, so both take the plain
  // decode path, and they must agree exactly.
  const json b = greedy(user_msg("Name the capital of France. One word."), 24);
  post(b);
  const std::string s2 = content_of(post(b)), s3 = content_of(post(b));
  check("greedy repeats exactly on the same path", !s2.empty() && s2 == s3,
        s2.empty() ? "empty output" : ("'" + s2.substr(0, 40) + "'"));

  // What is NOT a contract, and must not be asserted as one: greedy output is
  // not guaranteed identical BETWEEN the speculative and plain paths. The
  // acceptance rule is lossless, but the verify forward runs at T = block_size
  // through the tensor-core path while decode runs the GEMV at T = 1, and the
  // two sum in different orders -- so a pair of tokens whose logits sit within
  // a bf16 ulp of each other can swap. gate_spec measures this directly and
  // fails only when a divergence has a real logit gap. What this gate checks is
  // that the ESCAPE HATCH works, because a harness that needs reproducibility
  // across retries has to be able to ask for it.
  json ns = greedy(user_msg("Name three colours, comma separated."), 32);
  ns["speculative"] = false;
  const json r1 = post(ns), r2 = post(ns);
  check("speculative:false is honoured",
        r1.value("timings", json::object()).value("draft_rounds", 0) == 0, "");
  check("speculative:false repeats exactly",
        !content_of(r1).empty() && content_of(r1) == content_of(r2), "");
}

static void check_prefix_cache() {
  const std::string doc = long_prompt(1200, "The build id is ZX-4417.");
  json convo = user_msg(doc + "\n\nReply with the single word: ready.");
  const json t1 = post(greedy(convo, 16));
  if (!t1.contains("usage")) { bad("prefix cache turn 1", "no usage"); return; }
  const int cached1 = t1["usage"]["prompt_tokens_details"].value("cached_tokens", -1);
  const double pf1 = t1.value("timings", json::object()).value("prompt_per_second", 0.0);

  // Turn 2 extends the conversation, which is the shape an agent harness
  // actually produces: same prefix, more on the end.
  convo.push_back(json{{"role", "assistant"}, {"content", content_of(t1)}});
  convo.push_back(json{{"role", "user"}, {"content", "What is the build id? Answer with just the id."}});
  const json t2 = post(greedy(convo, 24));
  const int cached2 = t2["usage"]["prompt_tokens_details"].value("cached_tokens", -1);
  const double pf2 = t2.value("timings", json::object()).value("prompt_per_second", 0.0);
  const int prompt2 = t2["usage"].value("prompt_tokens", 0);

  check("turn 1 reports no cached tokens", cached1 == 0, "cached=" + std::to_string(cached1));
  check("turn 2 reuses most of the prefix", cached2 > prompt2 * 3 / 4,
        std::to_string(cached2) + " of " + std::to_string(prompt2));
  check("turn 2 prefill is faster", pf2 > pf1 * 2.0,
        std::to_string(int(pf1)) + " -> " + std::to_string(int(pf2)) + " tok/s");
  // The point of the cache is not speed for its own sake: the answer has to
  // come out of a context the model never re-read.
  const std::string a = content_of(t2);
  check("turn 2 answers from the reused context", a.find("ZX-4417") != std::string::npos,
        "'" + a.substr(0, 40) + "'");

  const std::string mt = get_metrics();
  check("prefix hits are visible in /metrics", metric(mt, "qwen_prefix_hits_total") > 0,
        "hits=" + std::to_string(int(metric(mt, "qwen_prefix_hits_total"))));
  check("cached_prefix_tokens counter advances",
        metric(mt, "qwen_cached_prefix_tokens_total") > 0, "");
}

static void check_speculation() {
  const json r = post(greedy(user_msg("Count from 1 to 30, comma separated."), 96));
  const json t = r.value("timings", json::object());
  const int rounds = t.value("draft_rounds", 0);
  const double acc = t.value("draft_accepted_per_round", 0.0);
  check("greedy request speculates", rounds > 0, "rounds=" + std::to_string(rounds));
  check("speculation commits more than 1/round", acc > 1.0,
        "accepted/round=" + std::to_string(acc));

  // A sampled request must NOT speculate: the acceptance rule is argmax
  // equality, so speculating a sampled request would silently change its
  // distribution.
  json sb = greedy(user_msg("Write one sentence about rivers."), 48);
  sb["temperature"] = 0.9; sb["top_p"] = 0.95; sb.erase("chat_template_kwargs");
  sb["chat_template_kwargs"] = {{"enable_thinking", false}};
  const json sr = post(sb);
  const int srounds = sr.value("timings", json::object()).value("draft_rounds", 0);
  check("sampled request does not speculate", srounds == 0,
        "rounds=" + std::to_string(srounds));

  const std::string mt = get_metrics();
  check("spec counters advance in /metrics",
        metric(mt, "qwen_spec_verify_steps_total") > 0 &&
        metric(mt, "qwen_spec_mean_acceptance_length") > 1.0,
        "mean_accept=" + std::to_string(metric(mt, "qwen_spec_mean_acceptance_length")));
}

static void check_seed_reproducibility() {
  // Accepting a `seed` is a contract: same request, same seed, same text. The
  // RNG is counter-based over (seed, step), so this fails if `step` is anything
  // other than the token index within THIS request.
  json b = greedy(user_msg("Write two sentences about a harbour at dawn."), 64);
  b["temperature"] = 0.9; b["top_p"] = 0.95; b["seed"] = 20260821;
  const std::string a1 = content_of(post(b)), a2 = content_of(post(b));
  check("same seed reproduces the same text", !a1.empty() && a1 == a2,
        a1.empty() ? "empty" : ("'" + a1.substr(0, 32) + "'"));

  json c = b; c["seed"] = 991;
  const std::string a3 = content_of(post(c));
  check("a different seed changes the text", !a3.empty() && a3 != a1, "");

  json d = b; d.erase("seed");
  const std::string a4 = content_of(post(d)), a5 = content_of(post(d));
  check("no seed still varies between requests", !a4.empty() && a4 != a5, "");
}

static void check_streaming() {
  const json msgs = user_msg("List three primary colours, comma separated.");
  // Both sides run with speculation off, so this compares the two RESPONSE
  // ENCODERS rather than two decode paths -- see check_greedy_determinism.
  json b = greedy(msgs, 48);
  b["speculative"] = false;
  const json nonstream = post(b);
  const std::string want = content_of(nonstream);

  auto c = mk(180);
  std::string acc, finish;
  bool saw_done = false, saw_usage = false;
  json sb = greedy(msgs, 48, true);
  sb["speculative"] = false;
  const std::string body = sb.dump();
  std::string buf;
  auto res = c.Post("/v1/chat/completions", httplib::Headers{}, body, "application/json",
                    [&](const char* data, size_t len) {
    buf.append(data, len);
    size_t p;
    while ((p = buf.find("\n\n")) != std::string::npos) {
      std::string ev = buf.substr(0, p);
      buf.erase(0, p + 2);
      const size_t d = ev.find("data: ");
      if (d == std::string::npos) continue;
      const std::string payload = ev.substr(d + 6);
      if (payload == "[DONE]") { saw_done = true; continue; }
      try {
        json j = json::parse(payload);
        if (j.contains("usage") && !j["usage"].is_null()) saw_usage = true;
        if (!j.contains("choices") || j["choices"].empty()) continue;
        const json& ch = j["choices"][0];
        if (ch.contains("delta")) {
          acc += ch["delta"].value("content", "");
          if (ch["delta"].contains("reasoning_content"))
            acc += ch["delta"].value("reasoning_content", "");
        }
        if (ch.contains("finish_reason") && !ch["finish_reason"].is_null())
          finish = ch["finish_reason"].get<std::string>();
      } catch (...) {}
    }
    return true;
  });
  check("stream terminates with [DONE]", res && saw_done, res ? "" : "transport error");
  check("stream carries a finish_reason", !finish.empty(), finish);
  check("stream_options.include_usage honoured", saw_usage, "");
  // The same greedy request, one prefix-cached and one not, must produce the
  // same text through two different response encoders.
  check("streamed text equals non-streamed", !want.empty() && acc == want,
        acc == want ? "" : ("'" + acc.substr(0, 30) + "' vs '" + want.substr(0, 30) + "'"));
}

static void check_tool_calls() {
  json b = greedy(user_msg("What is the weather in Paris? Use the tool."), 160);
  b["tools"] = json::array({json{
      {"type", "function"},
      {"function", {{"name", "get_weather"},
                    {"description", "Get the current weather for a city"},
                    {"parameters", {{"type", "object"},
                                    {"properties", {{"city", {{"type", "string"}}}}},
                                    {"required", json::array({"city"})}}}}}}});
  b["tool_choice"] = "auto";
  const json r = post(b);
  bool called = false, named = false, parsed = false;
  if (r.contains("choices") && !r["choices"].empty()) {
    const json& m = r["choices"][0]["message"];
    called = m.contains("tool_calls") && !m["tool_calls"].empty();
    if (called) {
      const json& f = m["tool_calls"][0]["function"];
      named = f.value("name", "") == "get_weather";
      try { parsed = json::parse(f.value("arguments", "")).contains("city"); } catch (...) {}
    }
  }
  check("tool call is emitted", called, "");
  check("tool call names the function", named, "");
  check("tool call arguments parse as JSON", parsed, "");
  if (called)
    check("finish_reason is tool_calls",
          r["choices"][0].value("finish_reason", "") == "tool_calls",
          r["choices"][0].value("finish_reason", ""));
}

// The actual agent loop: the model asks for a tool, the harness runs it and
// sends the result back as a `tool` message, and the model answers from it.
// A backend that emits tool calls but cannot READ tool results is useless to a
// coding harness, and nothing else in this repo exercises that direction.
static void check_tool_result_roundtrip() {
  const json tools = json::array({json{
      {"type", "function"},
      {"function", {{"name", "read_file"},
                    {"description", "Read a file from the repository"},
                    {"parameters", {{"type", "object"},
                                    {"properties", {{"path", {{"type", "string"}}}}},
                                    {"required", json::array({"path"})}}}}}}});
  json msgs = user_msg("Read the file VERSION.txt and tell me the version string it contains.");
  json b = greedy(msgs, 200);
  b["tools"] = tools;
  const json r1 = post(b);
  if (!r1.contains("choices") || !r1["choices"][0]["message"].contains("tool_calls")) {
    bad("tool result round trip", "model did not call the tool");
    return;
  }
  const json& tc = r1["choices"][0]["message"]["tool_calls"][0];
  msgs.push_back(r1["choices"][0]["message"]);
  msgs.push_back(json{{"role", "tool"},
                      {"tool_call_id", tc.value("id", "call_0")},
                      {"name", tc["function"].value("name", "read_file")},
                      {"content", "v7.3.1-beta"}});
  json b2 = greedy(msgs, 120);
  b2["tools"] = tools;
  const std::string a = content_of(post(b2));
  check("tool result round trip", a.find("7.3.1") != std::string::npos,
        "'" + a.substr(0, 48) + "'");
}

// Harnesses that predate the chat API use /v1/completions with a raw prompt.
static void check_text_completions() {
  int st = 0;
  const json r = post(json{{"model", "qwen38-27b"},
                           {"prompt", "The three primary colours are red, green and"},
                           {"temperature", 0}, {"max_tokens", 12}},
                      120, "/v1/completions", &st);
  bool shaped = false;
  std::string text;
  if (r.contains("choices") && !r["choices"].empty()) {
    shaped = r["choices"][0].contains("text");
    text = r["choices"][0].value("text", "");
  }
  check("/v1/completions responds", st == 200 && shaped, "status=" + std::to_string(st));
  // A raw completion has no chat template, so there is no enable_thinking to
  // set and the model may spend the whole budget reasoning. The contract is
  // that tokens were produced and accounted for, not that `text` is non-empty.
  const int n = r.contains("usage") ? r["usage"].value("completion_tokens", 0) : 0;
  check("/v1/completions generates", n > 0, std::to_string(n) + " tokens");
}

// Source code is not ASCII. A JSON or tokenizer round trip that mangles it
// corrupts the harness's diffs silently.
static void check_unicode_roundtrip() {
  const std::string s = "def f():\n    # \u00e9\u00e8 \u4e2d\u6587 \U0001F600 \u2014 na\u00efve\n    return '\u03b1\u03b2'";
  json msgs = user_msg(std::string("Repeat the following text back EXACTLY, with no commentary:\n") + s);
  const std::string a = content_of(post(greedy(msgs, 96)));
  // Not asserting an exact echo -- that is a model-capability question. What
  // must hold is that the bytes survive the transport: the reply is valid UTF-8
  // and the request did not error.
  bool valid = !a.empty();
  for (size_t i = 0; i < a.size() && valid;) {
    const unsigned char c = a[i];
    int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || i + n > a.size()) { valid = false; break; }
    for (int k = 1; k < n; ++k) if ((a[i + k] & 0xC0) != 0x80) { valid = false; break; }
    i += n;
  }
  check("non-ASCII prompt survives the round trip", valid, "'" + a.substr(0, 40) + "'");
}

static void check_stop_and_length() {
  json b = greedy(user_msg("Say: alpha bravo charlie delta"), 64);
  b["stop"] = json::array({"charlie"});
  const json r = post(b);
  const std::string s = content_of(r);
  check("stop sequence truncates", s.find("charlie") == std::string::npos,
        "'" + s.substr(0, 40) + "'");

  const json r1 = post(greedy(user_msg("Say yes."), 1));
  check("max_tokens=1 is not a special case",
        r1.contains("usage") && r1["usage"].value("completion_tokens", 0) == 1,
        r1.contains("usage") ? std::to_string(r1["usage"].value("completion_tokens", -1)) : "no usage");

  const json r2 = post(greedy(user_msg("Write a long paragraph about the sea."), 12));
  check("max_tokens gives finish_reason=length",
        r2.contains("choices") && r2["choices"][0].value("finish_reason", "") == "length",
        r2.contains("choices") ? r2["choices"][0].value("finish_reason", "") : "no choices");
  check("usage.completion_tokens honours max_tokens",
        r2.contains("usage") && r2["usage"].value("completion_tokens", 0) == 12,
        r2.contains("usage") ? std::to_string(r2["usage"].value("completion_tokens", -1)) : "");
}

static void check_bad_requests() {
  int st = 0;
  auto c = mk(30);
  auto r = c.Post("/v1/chat/completions", "{not json", "application/json");
  check("malformed JSON returns 4xx", r && r->status >= 400 && r->status < 500,
        r ? std::to_string(r->status) : "no response");

  post(json{{"model", "qwen38-27b"}}, 30, "/v1/chat/completions", &st);
  check("missing messages returns 4xx", st >= 400 && st < 500, std::to_string(st));

  // A prompt past --max-context must be refused, not truncated and not fatal.
  int max_ctx = 0;
  { auto mm = mk(20).Get("/v1/models");
    try { max_ctx = json::parse(mm->body)["data"][0].value("max_context", 0); } catch (...) {}
    if (max_ctx == 0) { auto h = mk(20).Get("/health");
      try { max_ctx = json::parse(h->body).value("max_context", 0); } catch (...) {} } }
  if (max_ctx > 0 && max_ctx <= 65536) {
    const std::string huge = long_prompt(max_ctx, "overflow");
    post(greedy(user_msg(huge), 8), 300, "/v1/chat/completions", &st);
    check("over-length prompt returns 4xx", st >= 400 && st < 500, std::to_string(st));
  } else {
    printf("  [skip] over-length prompt                    max_context %d too large to overflow cheaply\n",
           max_ctx);
  }
  check_liveness();   // still alive after the abuse
}

static void check_concurrency() {
  // The engine is a single slot behind a mutex. Concurrent requests must queue
  // and all complete correctly rather than interleave into each other's state.
  const int N = 4;
  std::vector<std::string> out(N);
  std::atomic<int> failed{0};
  std::vector<std::thread> th;
  for (int i = 0; i < N; ++i)
    th.emplace_back([&, i] {
      const json r = post(greedy(user_msg("What is " + std::to_string(10 + i) +
                                          " plus " + std::to_string(10 + i) +
                                          "? Answer with just the number."), 24), 300);
      out[i] = content_of(r);
      if (out[i].empty()) ++failed;
    });
  for (auto& t : th) t.join();
  check("concurrent requests all complete", failed == 0,
        std::to_string(N - failed) + "/" + std::to_string(N));
  int right = 0;
  for (int i = 0; i < N; ++i)
    if (out[i].find(std::to_string(2 * (10 + i))) != std::string::npos) ++right;
  check("concurrent answers are not crossed", right == N,
        std::to_string(right) + "/" + std::to_string(N) + " correct");
}

static void check_kv_and_retrieval(int words) {
  // A genuinely long prompt, decoded with the INT4 KV cache live. The needle is
  // near the front, so this fails if quantised keys lose the position.
  const std::string doc = long_prompt(words, "The deployment code is QT-9182.");
  const json r = post(greedy(user_msg(doc + "\n\nWhat is the deployment code? Answer with just the code."), 24), 900);
  const std::string a = content_of(r);
  const int n = r.contains("usage") ? r["usage"].value("prompt_tokens", 0) : 0;
  check("long-context retrieval through INT4 KV", a.find("QT-9182") != std::string::npos,
        std::to_string(n) + " tokens -> '" + a.substr(0, 30) + "'");
}

int main(int argc, char** argv) {
  int long_words = 8000;                  // ~10K tokens; bounded on purpose
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--host" && i + 1 < argc) g_host = argv[++i];
    else if (a == "--port" && i + 1 < argc) g_port = atoi(argv[++i]);
    else if (a == "--long-words" && i + 1 < argc) long_words = atoi(argv[++i]);
  }
  printf("gate_serving -> http://%s:%d\n\n", g_host.c_str(), g_port);

  printf("liveness\n");            check_liveness();
  printf("greedy determinism\n");  check_greedy_determinism();
  printf("prefix cache\n");        check_prefix_cache();
  printf("speculative decode\n");  check_speculation();
  printf("seeded sampling\n");     check_seed_reproducibility();
  printf("streaming\n");           check_streaming();
  printf("tool calls\n");          check_tool_calls();
  printf("agent loop\n");          check_tool_result_roundtrip();
  printf("/v1/completions\n");     check_text_completions();
  printf("unicode\n");             check_unicode_roundtrip();
  printf("stop / length\n");       check_stop_and_length();
  printf("malformed input\n");     check_bad_requests();
  printf("concurrency\n");         check_concurrency();
  printf("long context + INT4 KV\n"); check_kv_and_retrieval(long_words);

  printf("\n  %d passed, %d failed\n", g_pass, g_fail);
  return g_fail ? 1 : 0;
}
