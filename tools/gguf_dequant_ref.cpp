// ORACLE for the GGUF dequantisers: dumps a tensor dequantised by GGML ITSELF.
//
// llama.cpp is built in this tree, so rather than reimplement the reference and
// compare my reimplementation to my implementation, this links libggml and calls
// ggml_get_type_traits(type)->to_float -- the exact function llama.cpp runs. Any
// disagreement is then unambiguously ours.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include "../src/gguf/gguf.h"

extern "C" {
#include "ggml.h"
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("usage: %s FILE.gguf TENSOR OUT.f32 [max_elems]\n", argv[0]);
    return 2;
  }
  const int64_t cap = argc > 4 ? atoll(argv[4]) : 0;
  qwen::GgufFile f;
  try { f.open(argv[1]); }
  catch (const std::exception& e) { printf("FAIL: %s\n", e.what()); return 1; }

  const qwen::GgufTensor* t = f.find(argv[2]);
  if (!t) { printf("no tensor %s\n", argv[2]); return 1; }

  const int be = qwen::ggml_block_elems(t->type);
  int64_t n = int64_t(t->numel());
  if (cap > 0 && cap < n) n = (cap / be) * be;   // whole blocks only

  std::vector<float> out;
  out.resize(static_cast<size_t>(n));
  // ggml has no to_float trait for the unquantised types; they are trivial.
  if (t->type == qwen::GgmlType::F32) {
    memcpy(out.data(), t->data, size_t(n) * 4);
  } else if (t->type == qwen::GgmlType::F16) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(t->data);
    for (int64_t i = 0; i < n; ++i) out[size_t(i)] = ggml_fp16_to_fp32(h[i]);
  } else if (t->type == qwen::GgmlType::BF16) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(t->data);
    for (int64_t i = 0; i < n; ++i) {
      uint32_t u = uint32_t(h[i]) << 16;
      memcpy(&out[size_t(i)], &u, 4);
    }
  } else {
    const ggml_type_traits* tr = ggml_get_type_traits(ggml_type(uint32_t(t->type)));
    if (!tr || !tr->to_float) { printf("ggml has no to_float for %s\n",
                                       qwen::ggml_type_name(t->type)); return 1; }
    tr->to_float(t->data, out.data(), n);
  }

  std::ofstream o(argv[3], std::ios::binary);
  o.write(reinterpret_cast<const char*>(out.data()), out.size() * 4);
  printf("%s  type=%s  numel=%lld  dumped=%lld\n", argv[2],
         qwen::ggml_type_name(t->type), (long long)t->numel(), (long long)n);
  return 0;
}
