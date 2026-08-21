#pragma once
#include <string>
namespace qwen {
struct ServerConfig {
  std::string host = "127.0.0.1";
  int port = 8090;
  std::string model_alias = "qwen38-27b";
  int max_ctx = 32768;
  bool verbose = false;
};
}  // namespace qwen
