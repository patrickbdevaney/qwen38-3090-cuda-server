// GATE: the fused GGUF GEMV.
//
// Two levels, because they isolate different mistakes.
//
//  A. deq8 vs the full-block dequantiser, BIT-EXACT. The dequantiser is already
//     gated bit-exact against ggml, so this transitively pins the per-lane path
//     to ggml. It catches a wrong sub-block index, a wrong scale, a wrong
//     nibble half -- the mistakes that a dot product would average away.
//
//  B. the GEMV against a reference dot product built from those same values.
//     Not bit-exact: the kernel accumulates per lane and then reduces across the
//     warp, which is a different summation order from a sequential reference.
//     The bar is relative error, and it is tight because the values agree
//     exactly by (A).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cuda_runtime.h>
#include "../src/gguf/gguf.h"
#include "../src/gguf/dequant.cuh"
#include "../src/gguf/gemv.cuh"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)

static float b2f(uint16_t h){ uint32_t u=uint32_t(h)<<16; float f; memcpy(&f,&u,4); return f; }
static uint16_t f2b(float f){ uint32_t u; memcpy(&u,&f,4); return uint16_t(u>>16); }

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/gguf/Qwen3.8-27B-UD-Q3_K_XL.gguf";
  const int ROWS = argc > 2 ? atoi(argv[2]) : 256;   // output rows to check

  qwen::GgufFile f;
  try { f.open(path); }
  catch (const std::exception& e) { printf("cannot open: %s\n", e.what()); return 2; }

  // one representative 2-D tensor per type
  std::map<std::string, const qwen::GgufTensor*> by_type;
  for (const auto& kv : f.all()) {
    const auto& t = kv.second;
    if (t.ne.size() != 2 || t.row_len() % 256) continue;
    by_type.emplace(qwen::ggml_type_name(t.type), &t);
  }

  printf("gate_gguf_gemv: %s\n", path.c_str());
  printf("%-9s %-34s %8s %8s %12s %12s\n", "type", "tensor", "rows", "in_f",
         "deq8 bad", "gemv rel");
  bool ok = true;

  for (const auto& e : by_type) {
    const qwen::GgufTensor& t = *e.second;
    const int in_f = int(t.row_len());
    const int rows = std::min<int>(ROWS, int(t.rows()));
    if (!qwen::gguf_gemv_supported(t.type)) {
      printf("%-9s %-34s   NOT IMPLEMENTED\n", e.first.c_str(), t.name.c_str());
      ok = false;
      continue;
    }
    const size_t rb = qwen::gguf_row_bytes(t.type, in_f);
    const size_t nbytes = rb * size_t(rows);
    const size_t nelem = size_t(rows) * in_f;

    void* d_w = nullptr;
    CK(cudaMalloc(&d_w, nbytes));
    CK(cudaMemcpy(d_w, t.data, nbytes, cudaMemcpyHostToDevice));

    // ---- A: values ----
    float *d_ref = nullptr, *d_got = nullptr;
    CK(cudaMalloc(&d_ref, nelem * 4));
    CK(cudaMalloc(&d_got, nelem * 4));
    qwen::gguf_dequant_f32(d_ref, d_w, t.type, int64_t(nelem));
    qwen::gguf_deq8_dump(d_got, d_w, t.type, int64_t(nelem));
    CK(cudaDeviceSynchronize());
    std::vector<float> ref(nelem), got(nelem);
    CK(cudaMemcpy(ref.data(), d_ref, nelem * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got.data(), d_got, nelem * 4, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < nelem; ++i)
      if (memcmp(&ref[i], &got[i], 4) != 0) ++bad;

    // ---- B: the GEMV ----
    std::vector<uint16_t> x(in_f);
    uint32_t s = 99991;
    for (int i = 0; i < in_f; ++i) {
      s = s * 1664525u + 1013904223u;
      x[i] = f2b((float(s >> 8) / 8388608.0f - 1.0f) * 0.5f);
    }
    __nv_bfloat16 *d_x = nullptr, *d_y = nullptr;
    CK(cudaMalloc(&d_x, size_t(in_f) * 2));
    CK(cudaMalloc(&d_y, size_t(rows) * 2));
    CK(cudaMemcpy(d_x, x.data(), size_t(in_f) * 2, cudaMemcpyHostToDevice));
    qwen::GgufWeight w;
    w.data = d_w; w.type = t.type; w.out_f = rows; w.in_f = in_f;
    qwen::gguf_gemv(d_y, w, d_x);
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> y(rows);
    CK(cudaMemcpy(y.data(), d_y, size_t(rows) * 2, cudaMemcpyDeviceToHost));

    double se = 0, sw = 0;
    for (int r = 0; r < rows; ++r) {
      double acc = 0;
      for (int i = 0; i < in_f; ++i) acc += double(ref[size_t(r) * in_f + i]) * double(b2f(x[i]));
      const double d = double(b2f(y[r])) - acc;
      se += d * d; sw += acc * acc;
    }
    const double rel = sw > 0 ? std::sqrt(se / sw) : 0;

    // bf16 output has ~3 decimal digits, so 5e-3 relative is the storage floor.
    const bool tok = (bad == 0) && (rel < 5e-3);
    ok &= tok;
    printf("%-9s %-34s %8d %8d %12zu %12.3e   %s\n", e.first.c_str(), t.name.c_str(),
           rows, in_f, bad, rel, tok ? "OK" : "FAIL");

    cudaFree(d_w); cudaFree(d_ref); cudaFree(d_got); cudaFree(d_x); cudaFree(d_y);
  }

  size_t missing = 0;
  for (const auto& kv : f.all())
    if (kv.second.ne.size() == 2 && !qwen::gguf_gemv_supported(kv.second.type)) ++missing;
  if (missing) { printf("\n%zu weight tensors have an unimplemented type\n", missing); ok = false; }
  printf("\n  RESULT: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
