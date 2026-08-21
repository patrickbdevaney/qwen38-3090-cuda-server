// GATE: every GGUF dequantiser against GGML's own dequantize_row_*.
//
// The goldens come from tools/gguf_dequant_ref.cpp, which links llama.cpp's
// libggml and calls the exact function llama.cpp runs. So this is not "my
// reimplementation vs my implementation" -- a disagreement is unambiguously
// ours.
//
// The bar is BIT-EXACT. Both sides do the same float operations in the same
// order on the same integers; there is no rounding freedom to spend, and a
// tolerance here would hide a wrong shift or a wrong grid index that only shows
// up on rare values.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cuda_runtime.h>
#include "../src/gguf/gguf.h"
#include "../src/gguf/dequant.cuh"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)

int main(int argc, char** argv) {
  const std::string gguf = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/gguf/Qwen3.8-27B-UD-IQ4_XS.gguf";
  const std::string fx = argc > 2 ? argv[2] : "tests/fixtures/gguf";

  qwen::GgufFile f;
  try { f.open(gguf); }
  catch (const std::exception& e) { printf("cannot open %s: %s\n", gguf.c_str(), e.what()); return 2; }

  std::ifstream man(fx + "/manifest.txt");
  if (!man) { printf("no manifest in %s\n", fx.c_str()); return 2; }

  printf("gate_gguf: %s\n", gguf.c_str());
  printf("%-10s %-40s %10s %12s %12s\n", "type", "tensor", "elems", "mismatch", "max|d|");

  bool ok = true;
  size_t n_types = 0;
  std::string line;
  while (std::getline(man, line)) {
    std::istringstream ss(line);
    std::string type, name;
    long long n = 0;
    if (!(ss >> type >> name >> n)) continue;
    ++n_types;

    std::vector<float> want;
    want.resize(static_cast<size_t>(n));
    { std::ifstream b(fx + "/" + type + ".f32", std::ios::binary);
      if (!b) { printf("missing golden %s\n", type.c_str()); return 2; }
      b.read(reinterpret_cast<char*>(want.data()), want.size() * 4); }

    const qwen::GgufTensor& t = f.get(name);
    if (!qwen::gguf_dequant_supported(t.type)) {
      printf("%-10s %-40s %10lld   NOT IMPLEMENTED\n", type.c_str(), name.c_str(), n);
      ok = false;
      continue;
    }
    const int be = qwen::ggml_block_elems(t.type);
    const int bb = qwen::ggml_block_bytes(t.type);
    const size_t nblk = size_t(n) / size_t(be);

    void* d_src = nullptr;
    float* d_dst = nullptr;
    CK(cudaMalloc(&d_src, nblk * size_t(bb)));
    CK(cudaMalloc(&d_dst, size_t(n) * 4));
    CK(cudaMemcpy(d_src, t.data, nblk * size_t(bb), cudaMemcpyHostToDevice));
    qwen::gguf_dequant_f32(d_dst, d_src, t.type, n);
    CK(cudaDeviceSynchronize());
    std::vector<float> got;
    got.resize(static_cast<size_t>(n));
    CK(cudaMemcpy(got.data(), d_dst, got.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(d_src); cudaFree(d_dst);

    size_t bad = 0;
    double worst = 0;
    long nonfinite = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      if (!std::isfinite(got[i]) || !std::isfinite(want[i])) { ++nonfinite; continue; }
      if (memcmp(&got[i], &want[i], 4) != 0) {
        ++bad;
        worst = std::max(worst, std::fabs(double(got[i]) - double(want[i])));
      }
    }
    const bool tok = (bad == 0) && (nonfinite == 0);
    ok &= tok;
    printf("%-10s %-40s %10lld %12zu %12.3e   %s\n", type.c_str(), name.c_str(), n, bad,
           worst, tok ? "OK" : "FAIL");
  }

  // Every type in the file must be covered, or the loader would abort on a
  // tensor this gate never saw.
  size_t missing = 0;
  for (const auto& kv : f.all())
    if (!qwen::gguf_dequant_supported(kv.second.type)) ++missing;
  if (missing) {
    printf("\n%zu tensors have an unimplemented type\n", missing);
    ok = false;
  }
  printf("\n  %zu types checked, all bit-exact against ggml: %s\n", n_types,
         ok ? "yes" : "NO");
  printf("  RESULT: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
