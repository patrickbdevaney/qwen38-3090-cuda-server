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
};
}  // namespace qwen
