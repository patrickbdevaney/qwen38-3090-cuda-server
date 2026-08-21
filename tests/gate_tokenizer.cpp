// GATE (directive P1): the C++ tokenizer must reproduce HF exactly.
//
//   encode: our ids == HF ids, for every segment of the corpus
//   decode: our decode(HF ids) == HF's decoded text
//
// The corpus is >=100k tokens of mixed code / prose / CJK plus adversarial
// cases; fixtures come from tools/dump_tokenizer_ref.py and are committed, so
// this gate never invokes Python.
#include <cstdio>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include "../third_party/json.hpp"
#include "../src/tokenizer/bpe.h"

using json = nlohmann::json;

static std::string head(const std::string& s, size_t n = 80) {
  std::string o = s.substr(0, n);
  std::string e;
  for (char c : o) {
    if (c == '\n') e += "\\n";
    else if (c == '\t') e += "\\t";
    else if (c == '\r') e += "\\r";
    else e += c;
  }
  return e;
}

int main(int argc, char** argv) {
  const std::string tokjson = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ/tokenizer.json";
  const std::string corpus  = argc > 2 ? argv[2] : "tests/fixtures/tokenizer/corpus.jsonl";
  const std::string metaf   = argc > 3 ? argv[3] : "tests/fixtures/tokenizer/meta.json";

  qwen::Tokenizer tk;
  auto t0 = std::chrono::steady_clock::now();
  try { tk.load(tokjson); }
  catch (const std::exception& e) { fprintf(stderr, "load failed: %s\n", e.what()); return 2; }
  auto t1 = std::chrono::steady_clock::now();
  printf("loaded %zu base tokens in %.2f s\n", tk.base_vocab_size(),
         std::chrono::duration<double>(t1 - t0).count());

  // ---- special token ids must agree exactly ----
  size_t meta_fail = 0;
  {
    std::ifstream mf(metaf);
    if (mf) {
      json m; mf >> m;
      for (auto it = m["added_vocab"].begin(); it != m["added_vocab"].end(); ++it) {
        const int32_t want = it.value().get<int32_t>();
        const int32_t got  = tk.token_to_id(it.key());
        if (got != want) {
          if (++meta_fail <= 5)
            printf("  ADDED TOKEN MISMATCH %s: want %d got %d\n",
                   it.key().c_str(), want, got);
        }
      }
      const int32_t eos = m["eos_token_id"].get<int32_t>();
      if (tk.eos_id() != eos) { printf("  EOS MISMATCH want %d got %d\n", eos, tk.eos_id()); ++meta_fail; }
    }
  }

  std::ifstream f(corpus);
  if (!f) { fprintf(stderr, "cannot open %s\n", corpus.c_str()); return 2; }

  size_t nseg = 0, enc_fail = 0, dec_fail = 0, ntok = 0, nbytes = 0;
  double encode_s = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    json j = json::parse(line);
    const std::string name = j["name"].get<std::string>();
    const std::string text = j["text"].get<std::string>();
    const std::string wantdec = j["decoded"].get<std::string>();
    const std::vector<int32_t> want = j["ids"].get<std::vector<int32_t>>();

    auto a = std::chrono::steady_clock::now();
    const std::vector<int32_t> got = tk.encode(text);
    auto b = std::chrono::steady_clock::now();
    encode_s += std::chrono::duration<double>(b - a).count();

    ++nseg; ntok += want.size(); nbytes += text.size();

    if (got != want) {
      if (++enc_fail <= 8) {
        size_t k = 0;
        while (k < got.size() && k < want.size() && got[k] == want[k]) ++k;
        printf("  ENCODE MISMATCH [%s]  ours %zu ids, hf %zu ids, first diff at %zu\n",
               name.c_str(), got.size(), want.size(), k);
        auto ctx = [&](const std::vector<int32_t>& v) {
          std::string s;
          for (size_t q = (k > 3 ? k - 3 : 0); q < k + 4 && q < v.size(); ++q)
            s += (q == k ? " >>" : " ") + std::to_string(v[q]);
          return s;
        };
        printf("    hf  :%s\n    ours:%s\n", ctx(want).c_str(), ctx(got).c_str());
        std::vector<int32_t> around;
        for (size_t q = (k > 3 ? k - 3 : 0); q < k + 4 && q < want.size(); ++q)
          around.push_back(want[q]);
        printf("    hf text around diff: %s\n", head(tk.decode(around), 60).c_str());
      }
    }

    const std::string dec = tk.decode(want);
    if (dec != wantdec) {
      if (++dec_fail <= 5)
        printf("  DECODE MISMATCH [%s]\n    want %s\n    got  %s\n",
               name.c_str(), head(wantdec).c_str(), head(dec).c_str());
    }
  }

  printf("\ngate_tokenizer\n");
  printf("  segments        : %zu\n", nseg);
  printf("  tokens          : %zu\n", ntok);
  printf("  bytes           : %zu\n", nbytes);
  printf("  encode failures : %zu\n", enc_fail);
  printf("  decode failures : %zu\n", dec_fail);
  printf("  added-token/eos : %zu failures\n", meta_fail);
  printf("  encode speed    : %.2f MB/s (%.0f tok/s)\n",
         nbytes / encode_s / 1e6, ntok / encode_s);
  const bool pass = !enc_fail && !dec_fail && !meta_fail && ntok >= 100000;
  printf("  RESULT          : %s%s\n", pass ? "PASS" : "FAIL",
         ntok < 100000 ? "  (corpus below the 100k-token gate)" : "");
  return pass ? 0 : 1;
}
