// GATE 0 for the GGUF backend: parse the container and print what is in it.
// Same role tools/inspect_model.py plays for the safetensors checkpoint --
// derive every shape from the file rather than assuming one.
#include <cstdio>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include "../src/gguf/gguf.h"

int main(int argc, char** argv) {
  if (argc < 2) { printf("usage: %s FILE.gguf [--names]\n", argv[0]); return 2; }
  const bool names = argc > 2 && std::string(argv[2]) == "--names";
  qwen::GgufFile f;
  try { f.open(argv[1]); }
  catch (const std::exception& e) { printf("FAIL: %s\n", e.what()); return 1; }

  printf("gguf v%u, %zu tensors, %.2f GiB mapped\n", f.version(), f.all().size(),
         double(f.mapped_bytes()) / (1 << 30));
  printf("\n--- metadata ---\n");
  for (const auto& k : f.meta_keys()) {
    if (k.find("tokenizer.") == 0) continue;   // huge arrays
    const std::string s = f.meta_str(k);
    if (!s.empty()) { printf("  %-42s %s\n", k.c_str(), s.substr(0, 60).c_str()); continue; }
    printf("  %-42s %lld / %g\n", k.c_str(), (long long)f.meta_int(k, 0), f.meta_f64(k, 0));
  }

  std::map<std::string, std::pair<size_t, uint64_t>> by_type;  // count, bytes
  uint64_t total = 0;
  for (const auto& kv : f.all()) {
    const auto& t = kv.second;
    auto& e = by_type[qwen::ggml_type_name(t.type)];
    e.first++; e.second += t.nbytes;
    total += t.nbytes;
  }
  printf("\n--- tensor types ---\n");
  for (const auto& e : by_type)
    printf("  %-10s %4zu tensors %9.3f GiB  %5.1f%%\n", e.first.c_str(), e.second.first,
           double(e.second.second) / (1 << 30), 100.0 * double(e.second.second) / double(total));
  printf("  %-10s      %9.3f GiB\n", "TOTAL", double(total) / (1 << 30));

  if (names) {
    std::vector<std::string> ns;
    for (const auto& kv : f.all()) ns.push_back(kv.first);
    std::sort(ns.begin(), ns.end());
    printf("\n--- tensors ---\n");
    for (const auto& n : ns) {
      const auto& t = f.get(n);
      std::string d;
      for (size_t i = 0; i < t.ne.size(); ++i)
        d += (i ? "," : "") + std::to_string((long long)t.ne[i]);
      printf("  %-44s %-8s [%s]\n", n.c_str(), qwen::ggml_type_name(t.type), d.c_str());
    }
  }
  return 0;
}
