// Does verifying a block of M tokens cost the same weight read as decoding 1?
//
// That is the entire premise of speculative decoding on a bandwidth-bound
// model. If verify-8 costs 8x decode-1, speculation buys nothing.
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
  printf("%dx%d, %.1f MB of weights\n", W.out_f, W.in_f, bytes / 1e6);
  printf("%4s %10s %12s %10s %14s\n", "M", "ms", "GB/s", "vs M=1", "eff tok/s/layer");
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
    printf("%4d %10.3f %12.1f %9.2fx %14.1f\n", M, ms, bytes / (ms * 1e-3) / 1e9,
           ms / base, M / (ms * 1e-3) / 1000.0);
    cudaEventDestroy(a); cudaEventDestroy(b);
  }
  printf("\nIf verify-8 were 8x decode-1 there would be no point speculating.\n");
  qwen::awq_free(W); qwen::gemv_scratch_free(S);
  cudaFree(x); cudaFree(y);
  return 0;
}
