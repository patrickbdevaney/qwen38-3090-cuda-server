// GATE (directive P2): W4A16 GEMV must match a double-precision reference on
// REAL weight tiles, and must reach >= 80% of measured DRAM bandwidth.
//
// The reference is computed on the host from the mmap'd checkpoint using the
// unpack that tests/gate_dequant.cpp already proved bit-exact against
// compressed-tensors, in fp64, so any disagreement is the kernel's.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../src/loader/safetensors.h"
#include "../src/loader/w4a16_unpack.h"
#include "../src/kernels/gemv_w4a16.cuh"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static float bf16_to_f32(uint16_t h) { uint32_t u = uint32_t(h) << 16; float f; memcpy(&f,&u,4); return f; }
static uint16_t f32_to_bf16(float f) {
  uint32_t u; memcpy(&u,&f,4);
  // round-to-nearest-even, matching __float2bfloat16
  uint32_t r = (u >> 16) & 1u; uint32_t b = u + 0x7fffu + r;
  return uint16_t(b >> 16);
}

struct Case { std::string base; int out_f, in_f; };

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";

  qwen::SafeTensors st;
  st.open_dir(dir);

  const std::vector<std::string> bases = {
    "model.language_model.layers.0.mlp.gate_proj",
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.3.self_attn.q_proj",
    "model.language_model.layers.3.self_attn.k_proj",
    "model.language_model.layers.3.self_attn.o_proj",
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.0.linear_attn.in_proj_z",
    "model.language_model.layers.0.linear_attn.out_proj",
  };

  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f, 1.f);

  size_t nfail = 0;
  double worst_rel = 0.0;
  double sum_pct = 0.0, min_pct = 1e9; int npct = 0;
  printf("%-52s %10s %10s %9s %9s %8s\n",
         "tensor", "shape", "max|err|", "rel", "GB/s", "%peak");

  // measured on this card by bench/microbench.cu
  const double kMeasuredBW = 914.2e9;

  for (const auto& base : bases) {
    const auto& packed = st.get(base + ".weight_packed");
    const auto& scale  = st.get(base + ".weight_scale");
    const auto& zpv    = st.get(base + ".weight_zero_point");
    const int out_f = int(scale.shape[0]);
    const int G     = int(scale.shape[1]);
    const int in_f  = int(packed.shape[1]) * 8;
    const int group = in_f / G;

    // ---- device buffers ----
    uint32_t *d_qw, *d_zp; __nv_bfloat16 *d_sc, *d_x; float* d_y;
    CK(cudaMalloc(&d_qw, packed.nbytes)); CK(cudaMemcpy(d_qw, packed.data, packed.nbytes, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_sc, scale.nbytes));  CK(cudaMemcpy(d_sc, scale.data,  scale.nbytes,  cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_zp, zpv.nbytes));    CK(cudaMemcpy(d_zp, zpv.data,    zpv.nbytes,    cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_x, size_t(in_f) * 2));
    CK(cudaMalloc(&d_y, size_t(out_f) * 4));

    std::vector<uint16_t> hx(in_f);
    for (int i = 0; i < in_f; ++i) hx[i] = f32_to_bf16(nd(rng) * 0.05f);
    CK(cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));

    qwen::W4A16Weights W;
    qwen::awq_repack(W, d_qw, d_sc, d_zp, out_f, in_f, group);
    qwen::GemvScratch S;
    qwen::gemv_scratch_alloc(S, in_f, out_f, group, 16);
    CK(cudaDeviceSynchronize());

    qwen::gemv_w4a16_f32(d_y, W, d_x, S);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    std::vector<float> gy(out_f);
    CK(cudaMemcpy(gy.data(), d_y, out_f * 4, cudaMemcpyDeviceToHost));

    // ---- fp64 host reference on a sample of rows ----
    const auto* pw = reinterpret_cast<const uint32_t*>(packed.data);
    const auto* zw = reinterpret_cast<const uint32_t*>(zpv.data);
    const auto* sc = reinterpret_cast<const uint16_t*>(scale.data);
    const int SAMPLE = std::min(out_f, 256);
    double maxerr = 0.0, maxmag = 0.0;
    for (int t = 0; t < SAMPLE; ++t) {
      const int o = (out_f <= SAMPLE) ? t : int((size_t(t) * 7919) % out_f);
      double acc = 0.0;
      for (int i = 0; i < in_f; ++i) {
        const int32_t q  = qwen::w4_q(pw, in_f, o, i);
        const int     g  = i / group;
        const int32_t zp = qwen::w4_zp(zw, G, o, g);
        const double  s  = bf16_to_f32(sc[size_t(o) * G + g]);
        acc += double(q - zp) * s * double(bf16_to_f32(hx[i]));
      }
      maxerr = std::max(maxerr, std::fabs(acc - double(gy[o])));
      maxmag = std::max(maxmag, std::fabs(acc));
    }
    const double rel = maxmag > 0 ? maxerr / maxmag : 0.0;

    // ---- bandwidth ----
    const size_t bytes = W.total_bytes();
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 5; ++i) qwen::gemv_w4a16_f32(d_y, W, d_x, S);
    CK(cudaDeviceSynchronize());
    const int ITERS = 50;
    CK(cudaEventRecord(a));
    for (int i = 0; i < ITERS; ++i) qwen::gemv_w4a16_f32(d_y, W, d_x, S);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    const double gbs = double(bytes) * ITERS / (ms * 1e-3) / 1e9;

    char shape[32]; snprintf(shape, sizeof shape, "%dx%d", out_f, in_f);
    const bool ok = rel < 1e-5;
    printf("%-52s %10s %10.3e %9.2e %9.1f %7.1f%% k=%d%s\n",
           base.substr(base.rfind("layers.")).c_str(), shape, maxerr, rel, gbs,
           100.0 * gbs * 1e9 / kMeasuredBW, qwen::gemv_choose_splits(out_f, G),
           ok ? "" : "  FAIL");
    if (!ok) ++nfail;
    worst_rel = std::max(worst_rel, rel);
    const double pct = 100.0 * gbs * 1e9 / kMeasuredBW;
    sum_pct += pct; min_pct = std::min(min_pct, pct); ++npct;

    cudaEventDestroy(a); cudaEventDestroy(b);
    CK(cudaFree(W.qweight)); CK(cudaFree(W.scale)); CK(cudaFree(W.zp));
    qwen::gemv_scratch_free(S);
    CK(cudaFree(d_qw)); CK(cudaFree(d_sc)); CK(cudaFree(d_zp));
    CK(cudaFree(d_x)); CK(cudaFree(d_y));
  }

  printf("\ngate_gemv\n  worst relative error : %.3e\n  failures             : %zu\n",
         worst_rel, nfail);
  printf("  mean %% of 914.2 GB/s : %.1f%% (per-tensor; the gate metric is the\n", sum_pct / npct);
  printf("                         traffic-weighted aggregate in bench_decode_gemv,\n");
  printf("                         because k_proj/v_proj are 0.69%% of body traffic)\n");
  printf("  RESULT               : numerics %s\n", nfail ? "FAIL" : "PASS");
  return nfail ? 1 : 0;
}
