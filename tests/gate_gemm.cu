// GATE (directive P2): prefill GEMM numerics against an fp64 host reference on
// real weights, plus achieved TFLOPS against the 81.6 TFLOPS this card measures.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../src/loader/safetensors.h"
#include "../src/loader/w4a16_unpack.h"
#include "../src/kernels/gemm_w4a16.cuh"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static float bf16_to_f32(uint16_t h){ uint32_t u=uint32_t(h)<<16; float f; memcpy(&f,&u,4); return f; }
static uint16_t f32_to_bf16(float f){ uint32_t u; memcpy(&u,&f,4);
  uint32_t r=(u>>16)&1u, b=u+0x7fffu+r; return uint16_t(b>>16); }

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const int M = argc > 2 ? atoi(argv[2]) : 4096;   // the chunked-prefill chunk size

  qwen::SafeTensors st;
  st.open_dir(dir);

  const std::vector<std::string> bases = {
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.3.self_attn.q_proj",
    "model.language_model.layers.0.linear_attn.out_proj",
  };

  qwen::GemmWorkspace ws;
  qwen::gemm_workspace_alloc(ws, 256ull << 20);   // 256 MB weight tile buffer

  std::mt19937 rng(99);
  std::normal_distribution<float> nd(0.f, 1.f);
  size_t nfail = 0;
  double best_tflops = 0, sum_tflops = 0; int nt = 0;

  printf("M = %d\n%-40s %13s %11s %9s %9s\n", M, "tensor", "shape", "rel err", "TFLOPS", "%peak");
  for (const auto& base : bases) {
    const auto& packed = st.get(base + ".weight_packed");
    const auto& scale  = st.get(base + ".weight_scale");
    const auto& zpv    = st.get(base + ".weight_zero_point");
    const int N = int(scale.shape[0]), G = int(scale.shape[1]);
    const int K = int(packed.shape[1]) * 8, group = K / G;

    uint32_t *d_q, *d_z; __nv_bfloat16* d_s;
    CK(cudaMalloc(&d_q, packed.nbytes)); CK(cudaMemcpy(d_q, packed.data, packed.nbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_s, scale.nbytes));  CK(cudaMemcpy(d_s, scale.data, scale.nbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_z, zpv.nbytes));    CK(cudaMemcpy(d_z, zpv.data, zpv.nbytes, cudaMemcpyHostToDevice));
    qwen::W4A16Weights W;
    qwen::awq_repack(W, d_q, d_s, d_z, N, K, group);
    CK(cudaDeviceSynchronize());

    std::vector<uint16_t> hA(size_t(M) * K);
    for (auto& v : hA) v = f32_to_bf16(nd(rng) * 0.05f);
    __nv_bfloat16 *d_A, *d_C;
    CK(cudaMalloc(&d_A, hA.size() * 2)); CK(cudaMemcpy(d_A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_C, size_t(M) * N * 2));

    qwen::gemm_w4a16(d_C, W, d_A, M, ws);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    std::vector<uint16_t> hC(size_t(M) * N);
    CK(cudaMemcpy(hC.data(), d_C, hC.size() * 2, cudaMemcpyDeviceToHost));

    // fp64 reference on a sample of (m, n)
    const auto* pw = reinterpret_cast<const uint32_t*>(packed.data);
    const auto* zw = reinterpret_cast<const uint32_t*>(zpv.data);
    const auto* sc = reinterpret_cast<const uint16_t*>(scale.data);
    double maxerr = 0, maxmag = 0;
    for (int t = 0; t < 64; ++t) {
      const int m = int((size_t(t) * 7919) % M);
      const int n = int((size_t(t) * 104729) % N);
      double acc = 0;
      for (int k = 0; k < K; ++k) {
        const int32_t q  = qwen::w4_q(pw, K, n, k);
        const int     g  = k / group;
        const int32_t zp = qwen::w4_zp(zw, G, n, g);
        acc += double(q - zp) * bf16_to_f32(sc[size_t(n) * G + g]) * bf16_to_f32(hA[size_t(m) * K + k]);
      }
      maxerr = std::max(maxerr, std::fabs(acc - bf16_to_f32(hC[size_t(m) * N + n])));
      maxmag = std::max(maxmag, std::fabs(acc));
    }
    const double rel = maxmag > 0 ? maxerr / maxmag : 0;

    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) qwen::gemm_w4a16(d_C, W, d_A, M, ws);
    CK(cudaDeviceSynchronize());
    const int IT = 10;
    CK(cudaEventRecord(a));
    for (int i = 0; i < IT; ++i) qwen::gemm_w4a16(d_C, W, d_A, M, ws);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b)); ms /= IT;
    const double tf = 2.0 * M * N * K / (ms * 1e-3) / 1e12;

    char shp[32]; snprintf(shp, sizeof shp, "%dx%dx%d", M, N, K);
    // bf16 output rounds to 8 mantissa bits, so ~4e-3 relative is the floor here
    const bool ok = rel < 2e-2;
    printf("%-40s %13s %11.2e %9.1f %8.1f%%%s\n",
           base.substr(base.rfind("layers.")).c_str(), shp, rel, tf,
           100.0 * tf / 81.6, ok ? "" : "  FAIL");
    if (!ok) ++nfail;
    best_tflops = std::max(best_tflops, tf); sum_tflops += tf; ++nt;

    cudaEventDestroy(a); cudaEventDestroy(b);
    qwen::awq_free(W);
    CK(cudaFree(d_q)); CK(cudaFree(d_s)); CK(cudaFree(d_z));
    CK(cudaFree(d_A)); CK(cudaFree(d_C));
  }

  const double mean = sum_tflops / nt;
  printf("\ngate_gemm\n  mean %.1f TFLOPS (%.0f%% of 81.6 measured peak)\n", mean, 100 * mean / 81.6);
  printf("  projected prefill at 8K: %.0f tok/s   (llama.cpp measured 1318)\n",
         8192.0 / (2.0 * 24.33e9 * 8192 / (mean * 1e12)));
  printf("  RESULT: numerics %s\n", nfail ? "FAIL" : "PASS");
  qwen::gemm_workspace_free(ws);
  return nfail ? 1 : 0;
}
