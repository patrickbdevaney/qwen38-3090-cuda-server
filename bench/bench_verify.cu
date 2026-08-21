// Does verifying a block of M tokens cost the same weight read as decoding 1?
//
// That is the entire premise of speculative decoding on a bandwidth-bound
// model. If verify-8 costs 8x decode-1, speculation buys nothing.
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstring>
#include "../src/loader/safetensors.h"
#include "../src/kernels/gemv_w4a16.cuh"
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));return 1;}}while(0)

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  qwen::SafeTensors st; st.open_dir(dir);
  const int GROUP = 128;
  const std::string L0 = "model.language_model.layers.0.";

  // the dominant tensor: fused gate|up, 34816 x 5120
  std::vector<std::string> parts = {L0 + "mlp.gate_proj", L0 + "mlp.up_proj"};
  int total = 0;
  std::vector<std::tuple<uint32_t*, __nv_bfloat16*, uint32_t*, int>> src;
  for (auto& b : parts) {
    const auto& p = st.get(b + ".weight_packed");
    const auto& s = st.get(b + ".weight_scale");
    const auto& z = st.get(b + ".weight_zero_point");
    uint32_t *dp, *dz; __nv_bfloat16* ds;
    CK(cudaMalloc(&dp, p.nbytes)); CK(cudaMemcpy(dp, p.data, p.nbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&ds, s.nbytes)); CK(cudaMemcpy(ds, s.data, s.nbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dz, z.nbytes)); CK(cudaMemcpy(dz, z.data, z.nbytes, cudaMemcpyHostToDevice));
    src.push_back({dp, ds, dz, int(s.shape[0])});
    total += int(s.shape[0]);
  }
  qwen::W4A16Weights W;
  qwen::awq_alloc_fused(W, total, 5120, GROUP);
  int off = 0;
  for (auto& [dp, ds, dz, n] : src) { qwen::awq_repack_into(W, off, dp, ds, dz, n); off += n; }
  CK(cudaDeviceSynchronize());
  for (auto& [dp, ds, dz, n] : src) { cudaFree(dp); cudaFree(ds); cudaFree(dz); }

  qwen::GemvScratch S;
  qwen::gemv_scratch_alloc(S, W.in_f, W.out_f, GROUP, 16);
  __nv_bfloat16 *x, *y;
  CK(cudaMalloc(&x, size_t(16) * W.in_f * 2)); CK(cudaMemset(x, 0x3c, size_t(16) * W.in_f * 2));
  CK(cudaMalloc(&y, size_t(16) * W.out_f * 2));

  const double bytes = double(W.total_bytes());
  printf("%dx%d, %.1f MB of weights   (memory roofline %.3f ms at 914 GB/s)\n",
         W.out_f, W.in_f, bytes / 1e6, bytes / 914e9 * 1e3);

  // correctness of the MMA path against the scalar path, on the same inputs
  {
    const int M = 8;
    __nv_bfloat16* y2; CK(cudaMalloc(&y2, size_t(M) * W.out_f * 2));
    qwen::gemm_small_w4a16(y, W, x, M, S); CK(cudaDeviceSynchronize());
    qwen::gemm_mma_w4a16(y2, W, x, M, S);     CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::vector<uint16_t> a(size_t(M) * W.out_f), b2(a.size());
    CK(cudaMemcpy(a.data(), y, a.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b2.data(), y2, b2.size() * 2, cudaMemcpyDeviceToHost));
    auto f = [](uint16_t h){ uint32_t u = uint32_t(h) << 16; float v; memcpy(&v, &u, 4); return v; };
    double mx = 0, mag = 0; size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      if (!std::isfinite(f(b2[i]))) ++bad;
      mx = std::max(mx, std::fabs(double(f(a[i])) - f(b2[i])));
      mag = std::max(mag, std::fabs(double(f(a[i]))));
    }
    printf("MMA vs scalar at M=8: max|diff| %.3e, rel %.2e, non-finite %zu  -> %s\n",
           mx, mag > 0 ? mx / mag : 0.0, bad,
           (mag > 0 && mx / mag < 2e-2 && !bad) ? "OK" : "MISMATCH");
    cudaFree(y2);
  }

  printf("\n%4s %10s %12s %10s %12s %10s\n", "M", "scalar ms", "GB/s", "vs M=1", "MMA ms", "MMA x");
  double base = 0;
  for (int M : {1, 2, 4, 8, 16}) {
    for (int i = 0; i < 5; ++i) qwen::gemm_small_w4a16(y, W, x, M, S);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int IT = 30;
    cudaEventRecord(a);
    for (int i = 0; i < IT; ++i) qwen::gemm_small_w4a16(y, W, x, M, S);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= IT;
    if (!base) base = ms;
    for (int i = 0; i < 5; ++i) qwen::gemm_mma_w4a16(y, W, x, M, S);
    CK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < IT; ++i) qwen::gemm_mma_w4a16(y, W, x, M, S);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float mms = 0; cudaEventElapsedTime(&mms, a, b); mms /= IT;
    printf("%4d %10.3f %12.1f %9.2fx %12.3f %9.2fx\n", M, ms, bytes / (ms * 1e-3) / 1e9,
           ms / base, mms, mms / base);
    cudaEventDestroy(a); cudaEventDestroy(b);
  }
  printf("\nIf verify-8 were 8x decode-1 there would be no point speculating.\n");
  qwen::awq_free(W); qwen::gemv_scratch_free(S);
  cudaFree(x); cudaFree(y);
  return 0;
}
