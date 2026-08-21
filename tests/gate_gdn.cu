// GATE (directive P3): GatedDeltaNet must match the transformers reference at
// every stage, at 1 / 8 / 64 / 512 tokens, both from a fresh state and from a
// warm one.
//
// Goldens are per-primitive (conv, gates, scan, norm+gate) rather than
// whole-layer, so a failure localises to one stage instead of "the layer is
// wrong". Warm-state cases matter more than fresh ones: a fresh state is all
// zeros, which hides any bug in the decay term.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../third_party/json.hpp"
#include "../src/kernels/gdn.cuh"
#include "../src/loader/safetensors.h"

using json = nlohmann::json;
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static float b2f(uint16_t h){ uint32_t u=uint32_t(h)<<16; float f; memcpy(&f,&u,4); return f; }

template <typename T>
static std::vector<T> load(const std::string& p, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
  return v;
}

struct Stat { double maxabs = 0, maxrel = 0, ref_absmax = 0; };

static Stat cmp_bf16(const std::vector<uint16_t>& got, const std::vector<uint16_t>& want) {
  Stat s;
  for (size_t i = 0; i < want.size(); ++i) {
    const double a = b2f(got[i]), b = b2f(want[i]);
    s.maxabs = std::max(s.maxabs, std::fabs(a - b));
    s.ref_absmax = std::max(s.ref_absmax, std::fabs(b));
  }
  s.maxrel = s.ref_absmax > 0 ? s.maxabs / s.ref_absmax : 0;
  return s;
}
static Stat cmp_f32(const std::vector<float>& got, const std::vector<float>& want) {
  Stat s;
  for (size_t i = 0; i < want.size(); ++i) {
    s.maxabs = std::max(s.maxabs, std::fabs(double(got[i]) - double(want[i])));
    s.ref_absmax = std::max(s.ref_absmax, std::fabs(double(want[i])));
  }
  s.maxrel = s.ref_absmax > 0 ? s.maxabs / s.ref_absmax : 0;
  return s;
}

int main(int argc, char** argv) {
  const std::string fx = argc > 1 ? argv[1] : "tests/fixtures/gdn";
  const std::string wd = argc > 2 ? argv[2]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-BF16-from-INT4";

  json man; { std::ifstream f(fx + "/manifest.json"); f >> man; }
  qwen::GdnDims D;
  D.hidden = man["hidden"]; D.num_v_heads = man["num_v_heads"];
  D.num_k_heads = man["num_k_heads"]; D.head_k = man["head_k"];
  D.head_v = man["head_v"]; D.conv_k = man["conv_k"];
  D.rms_eps = man["rms_eps"].get<float>();
  const int CD = D.conv_dim(), VD = D.val_dim(), NV = D.num_v_heads, DV = D.head_v;

  // layer weights: conv1d, A_log, dt_bias, norm.weight -- read straight from the
  // bf16 checkpoint through the same loader the server uses
  qwen::SafeTensors st; st.open_dir(wd);
  const std::string P = "model.language_model.layers." +
                        std::to_string(man["layer"].get<int>()) + ".linear_attn.";
  auto up = [&](const std::string& n) {
    const auto& t = st.get(P + n);
    __nv_bfloat16* d; CK(cudaMalloc(&d, t.nbytes));
    CK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
    return d;
  };
  __nv_bfloat16* d_conv_w = up("conv1d.weight");
  __nv_bfloat16* d_Alog   = up("A_log");
  __nv_bfloat16* d_dt     = up("dt_bias");
  __nv_bfloat16* d_normw  = up("norm.weight");

  size_t nfail = 0;
  printf("%-10s %-18s %11s %11s %11s\n", "case", "stage", "max|diff|", "rel", "verdict");

  for (auto it = man["cases"].begin(); it != man["cases"].end(); ++it) {
    const std::string name = it.key();
    const int T = it.value()["T"];
    const bool warm = it.value()["seeded_state"];
    const std::string d = fx + "/" + name;

    auto h_mixed = load<uint16_t>(d + "/mixed_qkv.bf16", size_t(T) * CD);
    auto h_a     = load<uint16_t>(d + "/a.bf16", size_t(T) * NV);
    auto h_b     = load<uint16_t>(d + "/b.bf16", size_t(T) * NV);
    auto h_z     = load<uint16_t>(d + "/z.bf16", size_t(T) * VD);
    auto h_cs_in = load<float>(d + "/conv_state_in.f32", size_t(CD) * (D.conv_k - 1));
    auto h_S_in  = load<float>(d + "/state_in.f32", size_t(NV) * D.head_k * DV);

    auto r_qkv   = load<uint16_t>(d + "/qkv_postconv.bf16", size_t(T) * CD);
    auto r_g     = load<float>(d + "/g.f32", size_t(T) * NV);
    auto r_beta  = load<float>(d + "/beta.f32", size_t(T) * NV);
    auto r_core  = load<uint16_t>(d + "/core_attn_out.bf16", size_t(T) * VD);
    auto r_S_out = load<float>(d + "/state_out.f32", size_t(NV) * D.head_k * DV);
    auto r_gated = load<uint16_t>(d + "/gated.bf16", size_t(T) * VD);
    auto r_cs_out= load<float>(d + "/conv_state_out.f32", size_t(CD) * (D.conv_k - 1));

    __nv_bfloat16 *d_mixed, *d_qkv, *d_a, *d_b, *d_z, *d_core, *d_gated;
    float *d_cs, *d_S, *d_g, *d_beta;
    CK(cudaMalloc(&d_mixed, h_mixed.size()*2)); CK(cudaMemcpy(d_mixed, h_mixed.data(), h_mixed.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_a, h_a.size()*2)); CK(cudaMemcpy(d_a, h_a.data(), h_a.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_b, h_b.size()*2)); CK(cudaMemcpy(d_b, h_b.data(), h_b.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_z, h_z.size()*2)); CK(cudaMemcpy(d_z, h_z.data(), h_z.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_cs, h_cs_in.size()*4)); CK(cudaMemcpy(d_cs, h_cs_in.data(), h_cs_in.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_S, h_S_in.size()*4));   CK(cudaMemcpy(d_S, h_S_in.data(), h_S_in.size()*4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_qkv, size_t(T)*CD*2));
    CK(cudaMalloc(&d_core, size_t(T)*VD*2));
    CK(cudaMalloc(&d_gated, size_t(T)*VD*2));
    CK(cudaMalloc(&d_g, size_t(T)*NV*4));
    CK(cudaMalloc(&d_beta, size_t(T)*NV*4));

    qwen::gdn_conv(d_qkv, d_cs, d_mixed, d_conv_w, CD, D.conv_k, T, warm);
    qwen::gdn_gates(d_g, d_beta, d_a, d_b, d_Alog, d_dt, T, NV);
    qwen::gdn_scan(d_core, d_S, d_qkv, d_g, d_beta, D, T);
    qwen::gdn_norm_gate(d_gated, d_core, d_z, d_normw, T, NV, DV, D.rms_eps);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());

    auto fetch_b = [&](__nv_bfloat16* p, size_t n) {
      std::vector<uint16_t> v(n); CK(cudaMemcpy(v.data(), p, n*2, cudaMemcpyDeviceToHost)); return v; };
    auto fetch_f = [&](float* p, size_t n) {
      std::vector<float> v(n); CK(cudaMemcpy(v.data(), p, n*4, cudaMemcpyDeviceToHost)); return v; };

    struct Row { const char* stage; Stat s; double tol; };
    std::vector<Row> rows;
    rows.push_back({"conv+silu",  cmp_bf16(fetch_b(d_qkv, size_t(T)*CD), r_qkv), 8e-3});
    rows.push_back({"conv_state", cmp_f32 (fetch_f(d_cs, h_cs_in.size()), r_cs_out), 8e-3});
    rows.push_back({"g",          cmp_f32 (fetch_f(d_g, size_t(T)*NV), r_g), 1e-5});
    rows.push_back({"beta",       cmp_f32 (fetch_f(d_beta, size_t(T)*NV), r_beta), 1e-5});
    rows.push_back({"scan out",   cmp_bf16(fetch_b(d_core, size_t(T)*VD), r_core), 1.5e-2});
    rows.push_back({"state out",  cmp_f32 (fetch_f(d_S, h_S_in.size()), r_S_out), 1.5e-2});
    rows.push_back({"norm+gate",  cmp_bf16(fetch_b(d_gated, size_t(T)*VD), r_gated), 2e-2});

    for (auto& r : rows) {
      const bool ok = r.s.maxrel <= r.tol;
      if (!ok) ++nfail;
      printf("%-10s %-18s %11.3e %11.2e %11s\n", name.c_str(), r.stage,
             r.s.maxabs, r.s.maxrel, ok ? "ok" : "FAIL");
    }

    cudaFree(d_mixed); cudaFree(d_a); cudaFree(d_b); cudaFree(d_z);
    cudaFree(d_cs); cudaFree(d_S); cudaFree(d_qkv); cudaFree(d_core);
    cudaFree(d_gated); cudaFree(d_g); cudaFree(d_beta);
  }

  printf("\ngate_gdn: %zu stage failures -> %s\n", nfail, nfail ? "FAIL" : "PASS");
  return nfail ? 1 : 0;
}
