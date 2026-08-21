// qwen38-3090-cuda-server entry point.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "model/model.h"
#include "tokenizer/bpe.h"
#include "server/server_config.h"

namespace qwen { int run_server(Model&, Tokenizer&, const ServerConfig&); }

static void usage(const char* a0) {
  printf(
"usage: %s --model DIR [options]\n"
"\n"
"  --model DIR            checkpoint directory (compressed-tensors W4A16)\n"
"  --host HOST            default 127.0.0.1\n"
"  --port N               default 8090\n"
"  --alias NAME           model id reported by /v1/models (default qwen38-27b)\n"
"  --max-context N        default 32768; the server REFUSES TO START if this\n"
"                         does not fit with a 512 MB margin\n"
"  --prefill-chunk N      chunked-prefill chunk, default 4096 (84%% of peak)\n"
"  --lm-head-bits {16,8,4}  default 8. INT4 measured a KL of 1.8e-2 against\n"
"                         7.8e-4 for INT8 and is not recommended.\n"
"  --embed-bits {16,8}    default 8\n"
"  --webui PATH           single-file web UI served at / (default src/clients/webui/index.html)\n"
"  --draft DIR            DFlash2 drafter directory. Speculative decode is\n"
"                         GREEDY ONLY (the acceptance rule is argmax equality),\n"
"                         so sampled requests fall back to plain decode.\n"
"                         Measured 2.32x on greedy prompts.\n"
"  --draft-bf16           keep the drafter in bf16 (3.70 GiB) instead of W4A16\n"
"                         (1.25 GiB). Measured slower and no more accurate.\n"
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
    else if (a == "--max-context") opt.max_ctx = atoi(next());
    else if (a == "--prefill-chunk") opt.max_batch = atoi(next());
    else if (a == "--lm-head-bits") opt.lm_head_bits = atoi(next());
    else if (a == "--embed-bits") opt.quantize_embed = atoi(next()) == 8;
    else if (a == "--webui") cfg.webui_path = next();
    else if (a == "--draft") cfg.draft_dir = next();
    else if (a == "--draft-bf16") cfg.draft_quantize = false;
    else if (a == "--prefix-slots") cfg.prefix_slots = atoi(next());
    else if (a == "--no-prefix-cache") cfg.prefix_cache = false;
    else if (a == "--no-graph") no_graph = true;
    else if (a == "--bench") bench = true;
    else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
    else { fprintf(stderr, "unknown option %s\n", a.c_str()); usage(argv[0]); return 2; }
  }
  if (model_dir.empty()) { usage(argv[0]); return 2; }
  cfg.max_ctx = opt.max_ctx;

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
