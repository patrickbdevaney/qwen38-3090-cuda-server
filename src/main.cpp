// qwen38-3090-cuda-server entry point.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "model/model.h"
#include "tokenizer/bpe.h"
#include "server/server_config.h"

namespace qwen { int run_server(Model&, Tokenizer&, const ServerConfig&); }
namespace qwen { extern std::string g_model_dir; }

static void usage(const char* a0) {
  printf(
"usage: %s --model DIR [options]\n"
"\n"
"  --model DIR            checkpoint directory (compressed-tensors W4A16)\n"
"  --host HOST            default 127.0.0.1\n"
"  --port N               default 8090\n"
"  --alias NAME           model id reported by /v1/models (default qwen38-27b)\n"
"  --max-context N|auto   default 32768. `auto` gives the KV cache everything\n"
"                         left after the model and workspaces, capped at the\n"
"                         model's trained 262144 and keeping a 512 MB margin.\n"
"                         With a number, the server REFUSES TO START if it does\n"
"                         not fit with that margin.\n"
"  --prefill-chunk N      chunked-prefill chunk, default 4096 (84%% of peak)\n"
"  --lm-head-bits {16,8,4}  default 8. INT4 measured a KL of 1.8e-2 against\n"
"                         7.8e-4 for INT8 and is not recommended.\n"
"  --embed-bits {16,8}    default 8\n"
"  --kv-cache FMT         KV cache format: fp8 (default), v4 (K fp8, V int4),\n"
"                         int4 (both). FP8 is 32 KiB/token across the 16\n"
"                         attention layers; int4 with a per-32-group scale is\n"
"                         18 KiB, which at 262144 context is 4.5 GiB instead of\n"
"                         8.0. Keys decide WHERE attention goes and values only\n"
"                         what it carries, so v4 is the conservative step.\n"
"  --embed-host           keep embed_tokens in HOST memory and DMA one row per\n"
"                         token. Reclaims 1.185 GiB of device memory -- 38k\n"
"                         tokens of FP8 KV -- at no accuracy cost, because the\n"
"                         embedding is a gather and never a matmul. Required to\n"
"                         reach the model's native 262144 context on 24 GB.\n"
"  --webui PATH           single-file web UI served at / (default src/clients/webui/index.html)\n"
"  --draft DIR            DFlash2 drafter directory. Speculative decode is\n"
"                         GREEDY ONLY (the acceptance rule is argmax equality),\n"
"                         so sampled requests fall back to plain decode.\n"
"                         Measured 2.32x on greedy prompts.\n"
"  --draft-bf16           keep the drafter in bf16 (3.70 GiB) instead of W4A16\n"
"                         (1.25 GiB). Measured slower and no more accurate.\n"
"  --vision               load the vision tower and accept image content parts.\n"
"                         Costs 0.858 GiB resident, which is 28,114 tokens of\n"
"                         FP8 KV, so it is off by default. Images arrive as\n"
"                         data: URLs or local paths; remote URLs are not fetched.\n"
"  --vision-max-patches N largest image, in 16px patches, default 4096\n"
"                         (1024 image tokens after the 2x2 merge)\n"
"  --prefix-slots N       recurrent-state snapshots kept for prefix reuse\n"
"                         (default 4). Each is ~150 MiB of PINNED HOST memory\n"
"                         and zero device memory. Measured 46x faster prefill\n"
"                         on the second turn of a conversation.\n"
"  --no-prefix-cache      disable it\n"
"  --no-graph             disable CUDA graph capture (about 7%% slower)\n"
"  --bench                run a decode benchmark instead of serving\n"
"  -h, --help\n", a0);
}

int main(int argc, char** argv) {
  std::string model_dir;
  qwen::ServerConfig cfg;
  qwen::LoadOptions opt;
  opt.max_ctx = 32768;
  bool bench = false, no_graph = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); }
      return argv[++i];
    };
    if (a == "--model") model_dir = next();
    else if (a == "--host") cfg.host = next();
    else if (a == "--port") cfg.port = atoi(next());
    else if (a == "--alias") cfg.model_alias = next();
    else if (a == "--max-context") {
      const std::string v = next();
      opt.max_ctx = (v == "auto") ? 0 : atoi(v.c_str());
    }
    else if (a == "--prefill-chunk") opt.max_batch = atoi(next());
    else if (a == "--lm-head-bits") opt.lm_head_bits = atoi(next());
    else if (a == "--embed-bits") opt.quantize_embed = atoi(next()) == 8;
    else if (a == "--kv-cache") {
      const std::string v = next();
      if (v == "fp8")       { opt.kv_k = qwen::KvFmt::FP8;  opt.kv_v = qwen::KvFmt::FP8; }
      else if (v == "v4")   { opt.kv_k = qwen::KvFmt::FP8;  opt.kv_v = qwen::KvFmt::INT4; }
      else if (v == "int4") { opt.kv_k = qwen::KvFmt::INT4; opt.kv_v = qwen::KvFmt::INT4; }
      else { fprintf(stderr, "--kv-cache must be fp8, v4 or int4\n"); return 2; }
    }
    else if (a == "--embed-host") opt.embed_host = true;
    else if (a == "--webui") cfg.webui_path = next();
    else if (a == "--draft") cfg.draft_dir = next();
    else if (a == "--draft-bf16") cfg.draft_quantize = false;
    else if (a == "--vision") cfg.vision = true;
    else if (a == "--vision-max-patches") cfg.vision_max_patches = atoi(next());
    else if (a == "--prefix-slots") cfg.prefix_slots = atoi(next());
    else if (a == "--no-prefix-cache") cfg.prefix_cache = false;
    else if (a == "--no-graph") no_graph = true;
    else if (a == "--bench") bench = true;
    else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(argv[0]); return 2; }
  }
  if (model_dir.empty()) { usage(argv[0]); return 2; }
  cfg.max_ctx = opt.max_ctx;

  qwen::g_model_dir = model_dir;
  qwen::Model m;
  qwen::model_load(m, model_dir, opt);

  qwen::Tokenizer tok;
  tok.load(model_dir + "/tokenizer.json");
  printf("tokenizer: %zu tokens\n", tok.base_vocab_size());

  if (!no_graph) qwen::model_graph_capture(m);
  m.use_graph = !no_graph;

  if (bench) {
    printf("(use bench/bench_decode for the full curve)\n");
    return 0;
  }
  return qwen::run_server(m, tok, cfg);
}
