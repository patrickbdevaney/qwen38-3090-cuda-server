#pragma once
#include <string>
namespace qwen {
struct ServerConfig {
  std::string host = "127.0.0.1";
  int port = 8090;
  std::string model_alias = "qwen38-27b";
  int max_ctx = 32768;
  bool verbose = false;
  std::string webui_path = "src/clients/webui/index.html";
  // DFlash2 drafter. Speculation is greedy-only, because the acceptance rule is
  // argmax equality; a sampled request falls back to plain decode.
  std::string draft_dir;
  bool draft_quantize = true;

  // Prefix cache. Snapshots live in pinned host memory, so slots cost host RAM
  // and no device memory at all.
  int  prefix_slots = 4;
  bool prefix_cache = true;

  // Vision. Costs 0.858 GiB resident -- 28,114 tokens of FP8 KV -- which is why
  // it is opt-in rather than always loaded.
  bool vision = false;
  int  vision_max_patches = 4096;   // 1024 image tokens after the 2x2 merge
};
}  // namespace qwen
