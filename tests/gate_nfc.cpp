// GATE: NFC normalization must match Python's unicodedata exactly.
//
// This is a prerequisite for the tokenizer gate, not an optional extra: the
// Qwen tokenizer's normalizer is NFC, so a single disagreement here changes
// token ids for every prompt containing that sequence.
#include <cstdio>
#include <fstream>
#include <string>
#include "../third_party/json.hpp"
#include "../src/tokenizer/unicode.h"

using json = nlohmann::json;

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "tests/fixtures/nfc_cases.jsonl";
  std::ifstream f(path);
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return 2; }

  size_t n = 0, fail = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    json j = json::parse(line);
    const std::string in   = j["in"].get<std::string>();
    const std::string want = j["nfc"].get<std::string>();
    const std::string got  = qwen::nfc(in);
    ++n;
    if (got != want) {
      if (++fail <= 12) {
        auto hex = [](const std::string& s) {
          std::string o;
          for (size_t i = 0; i < s.size();) {
            uint32_t cp = qwen::utf8_next(s, i);
            char b[16]; snprintf(b, sizeof b, "U+%04X ", cp); o += b;
          }
          return o;
        };
        printf("  MISMATCH #%zu\n    in   %s\n    want %s\n    got  %s\n",
               fail, hex(in).c_str(), hex(want).c_str(), hex(got).c_str());
      }
    }
  }
  printf("gate_nfc: %zu cases, %zu mismatches -> %s\n", n, fail,
         fail ? "FAIL" : "PASS");
  return fail ? 1 : 0;
}
