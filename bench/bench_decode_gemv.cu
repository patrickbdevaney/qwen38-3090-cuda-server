// Decode-step GEMV benchmark: the number that actually predicts tok/s.
//
// Per-tensor efficiency is a diagnostic, not the gate. What sets decode speed is
// the AGGREGATE effective bandwidth over one token's worth of GEMVs, weighted by
// the bytes each one moves. 84.8% of the body's weight traffic sits in the large
// mlp and qkv projections; k_proj and v_proj together are 0.69%, so their poor
// standalone efficiency is nearly irrelevant -- and fusing them away removes it.
//
// Fusions applied here are the ones the real model will use, because the inputs
// are genuinely shared:
//   attention : q|k|v  -> [14336, 5120]     (all read the same normed hidden)
//   mlp       : gate|up -> [34816, 5120]    (both read the post-attention norm)
//   gdn       : in_proj_qkv|in_proj_z -> [16384, 5120]
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../src/loader/safetensors.h"
#include "../src/kernels/gemv_w4a16.cuh"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

struct Src { const void* q; const void* s; const void* z; int out_f; };

struct Op {
  std::string name;
  qwen::W4A16Weights w;
  int repeats;            // how many times this shape appears per decoded token
};

static Src get(qwen::SafeTensors& st, const std::string& base) {
  return {st.get(base + ".weight_packed").data, st.get(base + ".weight_scale").data,
          st.get(base + ".weight_zero_point").data,
          int(st.get(base + ".weight_scale").shape[0])};
}

static void upload(const qwen::SafeTensors& st, const std::string& base,
                   uint32_t** dq, __nv_bfloat16** ds, uint32_t** dz) {
  const auto& p = st.get(base + ".weight_packed");
  const auto& s = st.get(base + ".weight_scale");
  const auto& z = st.get(base + ".weight_zero_point");
  CK(cudaMalloc(dq, p.nbytes)); CK(cudaMemcpy(*dq, p.data, p.nbytes, cudaMemcpyHostToDevice));
  CK(cudaMalloc(ds, s.nbytes)); CK(cudaMemcpy(*ds, s.data, s.nbytes, cudaMemcpyHostToDevice));
  CK(cudaMalloc(dz, z.nbytes)); CK(cudaMemcpy(*dz, z.data, z.nbytes, cudaMemcpyHostToDevice));
}

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const bool fused = !(argc > 2 && std::string(argv[2]) == "--unfused");

  qwen::SafeTensors st;
  st.open_dir(dir);
  const int GROUP = 128;

  std::vector<Op> ops;
  auto add_fused = [&](const std::string& name, std::vector<std::string> parts,
                       int in_f, int repeats) {
    int total = 0;
    std::vector<Src> srcs;
    std::vector<std::tuple<uint32_t*, __nv_bfloat16*, uint32_t*>> dev;
    for (auto& b : parts) {
      uint32_t *dq, *dz; __nv_bfloat16* ds;
      upload(st, b, &dq, &ds, &dz);
      const int of = int(st.get(b + ".weight_scale").shape[0]);
      dev.push_back({dq, ds, dz});
      srcs.push_back({dq, ds, dz, of});
      total += of;
    }
    qwen::W4A16Weights w;
    qwen::awq_alloc_fused(w, total, in_f, GROUP);
    int off = 0;
    for (size_t i = 0; i < srcs.size(); ++i) {
      qwen::awq_repack_into(w, off, (const uint32_t*)srcs[i].q,
                            (const __nv_bfloat16*)srcs[i].s,
                            (const uint32_t*)srcs[i].z, srcs[i].out_f);
      off += srcs[i].out_f;
    }
    CK(cudaDeviceSynchronize());
    for (auto& [a, b2, c] : dev) { cudaFree(a); cudaFree(b2); cudaFree(c); }
    ops.push_back({name, w, repeats});
  };

  const std::string L0 = "model.language_model.layers.0.";
  const std::string L3 = "model.language_model.layers.3.";

  // 48 GatedDeltaNet layers + 16 full-attention layers, all with the same MLP.
  if (fused) {
    add_fused("gdn.in_proj_qkv|z", {L0 + "linear_attn.in_proj_qkv", L0 + "linear_attn.in_proj_z"}, 5120, 48);
    add_fused("gdn.out_proj",      {L0 + "linear_attn.out_proj"},   6144, 48);
    add_fused("attn.q|k|v",        {L3 + "self_attn.q_proj", L3 + "self_attn.k_proj", L3 + "self_attn.v_proj"}, 5120, 16);
    add_fused("attn.o_proj",       {L3 + "self_attn.o_proj"},       6144, 16);
    add_fused("mlp.gate|up",       {L0 + "mlp.gate_proj", L0 + "mlp.up_proj"}, 5120, 64);
    add_fused("mlp.down",          {L0 + "mlp.down_proj"},          17408, 64);
  } else {
    add_fused("gdn.in_proj_qkv", {L0 + "linear_attn.in_proj_qkv"}, 5120, 48);
    add_fused("gdn.in_proj_z",   {L0 + "linear_attn.in_proj_z"},   5120, 48);
    add_fused("gdn.out_proj",    {L0 + "linear_attn.out_proj"},    6144, 48);
    add_fused("attn.q_proj",     {L3 + "self_attn.q_proj"},        5120, 16);
    add_fused("attn.k_proj",     {L3 + "self_attn.k_proj"},        5120, 16);
    add_fused("attn.v_proj",     {L3 + "self_attn.v_proj"},        5120, 16);
    add_fused("attn.o_proj",     {L3 + "self_attn.o_proj"},        6144, 16);
    add_fused("mlp.gate_proj",   {L0 + "mlp.gate_proj"},           5120, 64);
    add_fused("mlp.up_proj",     {L0 + "mlp.up_proj"},             5120, 64);
    add_fused("mlp.down",        {L0 + "mlp.down_proj"},           17408, 64);
  }

  int max_in = 0, max_out = 0;
  for (auto& o : ops) { max_in = std::max(max_in, o.w.in_f); max_out = std::max(max_out, o.w.out_f); }
  qwen::GemvScratch S;
  qwen::gemv_scratch_alloc(S, max_in, max_out, GROUP, 16);

  __nv_bfloat16 *d_x, *d_y;
  CK(cudaMalloc(&d_x, size_t(max_in) * 2));
  CK(cudaMalloc(&d_y, size_t(max_out) * 2));
  CK(cudaMemset(d_x, 0x3c, size_t(max_in) * 2));

  const double kBW = 914.2e9;
  printf("%-22s %14s %8s %9s %9s %7s %6s\n",
         "op", "shape", "xN", "MB each", "GB/s", "%peak", "k");
  double total_bytes = 0, total_ms = 0;
  for (auto& o : ops) {
    for (int i = 0; i < 5; ++i) qwen::gemv_w4a16(d_y, o.w, d_x, S);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    const int IT = 30;
    CK(cudaEventRecord(a));
    for (int i = 0; i < IT; ++i) qwen::gemv_w4a16(d_y, o.w, d_x, S);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
    ms /= IT;
    const double bytes = double(o.w.total_bytes());
    const double gbs = bytes / (ms * 1e-3) / 1e9;
    char shape[32]; snprintf(shape, sizeof shape, "%dx%d", o.w.out_f, o.w.in_f);
    printf("%-22s %14s %8d %9.1f %9.1f %6.1f%% %6d\n", o.name.c_str(), shape,
           o.repeats, bytes / 1e6, gbs, 100.0 * gbs * 1e9 / kBW,
           qwen::gemv_choose_splits(o.w.out_f, o.w.num_groups));
    total_bytes += bytes * o.repeats;
    total_ms    += double(ms) * o.repeats;
    cudaEventDestroy(a); cudaEventDestroy(b);
  }

  // lm_head, once per token, quantized to INT4 g128 by our repack (0.615 GiB)
  const double lm_head_bytes = 248320.0 * 5120 / 2 + 248320.0 * 40 * 3;
  const double body_gbs = total_bytes / (total_ms * 1e-3) / 1e9;
  const double lm_ms = lm_head_bytes / (0.80 * kBW) * 1e3;   // assume 80%, measured later
  const double step_ms = total_ms + lm_ms;

  printf("\n=== decode step (body only, %s) ===\n", fused ? "fused" : "unfused");
  printf("  weight traffic     : %.3f GiB\n", total_bytes / 1073741824.0);
  printf("  time               : %.2f ms\n", total_ms);
  printf("  AGGREGATE BANDWIDTH: %.1f GB/s = %.1f%% of measured 914.2 GB/s\n",
         body_gbs, 100.0 * body_gbs * 1e9 / kBW);
  printf("  body-only ceiling  : %.1f tok/s\n", 1000.0 / total_ms);
  printf("\n  + lm_head INT4 %.3f GiB at an assumed 80%%: %.2f ms\n",
         lm_head_bytes / 1073741824.0, lm_ms);
  printf("  PROJECTED AR DECODE: %.1f tok/s   (llama.cpp measured 38.41)\n",
         1000.0 / step_ms);

  const double pct = 100.0 * body_gbs * 1e9 / kBW;
  printf("\n  GATE (P2, >= 80%% of measured DRAM bandwidth): %s\n",
         pct >= 80.0 ? "PASS" : "FAIL");

  for (auto& o : ops) qwen::awq_free(o.w);
  qwen::gemv_scratch_free(S);
  return pct >= 80.0 ? 0 : 1;
}
