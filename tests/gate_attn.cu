// GATE (directive P4): gated attention against the transformers reference at
// 1 / 8 / 128 tokens AND at positions 0, 1, 4095, 32767, 131071.
//
// The long-position cases are not padding. A wrong mrope produces output that is
// fluent at short context and decays at long context -- the directive's failure
// mode #1 -- so the rope table is checked where it actually matters, not only
// near zero.
//
// The reference is dumped twice: with bf16 KV (what HF does) and with KV
// round-tripped through float8_e4m3fn (what we do). The kernel is gated against
// the fp8 variant, so kernel correctness is isolated from the cost of the FP8
// cache; that cost is reported separately.
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "../third_party/json.hpp"
#include "../src/kernels/attn.cuh"
#include "../src/loader/safetensors.h"

using json = nlohmann::json;
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static float b2f(uint16_t h){ uint32_t u=uint32_t(h)<<16; float f; memcpy(&f,&u,4); return f; }

template <typename T> static std::vector<T> load(const std::string& p, size_t n) {
  std::vector<T> v(n);
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "missing %s\n", p.c_str()); exit(2); }
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
  return v;
}
struct Stat { double maxabs = 0, rel = 0; };
static Stat cmp_b(const std::vector<uint16_t>& g, const std::vector<uint16_t>& w) {
  Stat s; double mx = 0;
  for (size_t i = 0; i < w.size(); ++i) {
    s.maxabs = std::max(s.maxabs, std::fabs(double(b2f(g[i])) - b2f(w[i])));
    mx = std::max(mx, std::fabs(double(b2f(w[i]))));
  }
  s.rel = mx > 0 ? s.maxabs / mx : 0; return s;
}
static Stat cmp_f(const std::vector<float>& g, const std::vector<float>& w) {
  Stat s; double mx = 0;
  for (size_t i = 0; i < w.size(); ++i) {
    s.maxabs = std::max(s.maxabs, std::fabs(double(g[i]) - w[i]));
    mx = std::max(mx, std::fabs(double(w[i])));
  }
  s.rel = mx > 0 ? s.maxabs / mx : 0; return s;
}

int main(int argc, char** argv) {
  const std::string fx = argc > 1 ? argv[1] : "tests/fixtures/attn";
  const std::string wd = argc > 2 ? argv[2]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-BF16-from-INT4";

  json man; { std::ifstream f(fx + "/manifest.json"); f >> man; }
  qwen::AttnDims D;
  D.num_q_heads = man["num_q_heads"]; D.num_kv_heads = man["num_kv_heads"];
  D.head_dim = man["head_dim"];       D.rotary_dim = man["rotary_dim"];
  D.rms_eps = man["rms_eps"].get<float>();
  D.rope_theta = man["rope_theta"].get<float>();
  const int NQ = D.num_q_heads, NKV = D.num_kv_heads, HD = D.head_dim;
  const int HALF = D.rotary_dim / 2;

  qwen::SafeTensors st; st.open_dir(wd);
  const std::string P = "model.language_model.layers." +
                        std::to_string(man["layer"].get<int>()) + ".self_attn.";
  auto up = [&](const std::string& n) {
    const auto& t = st.get(P + n);
    __nv_bfloat16* d; CK(cudaMalloc(&d, t.nbytes));
    CK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice)); return d; };
  __nv_bfloat16* d_qn = up("q_norm.weight");
  __nv_bfloat16* d_kn = up("k_norm.weight");

  cublasHandle_t cb; cublasCreate(&cb);
  size_t nfail = 0;
  printf("%-12s %-16s %11s %11s %8s\n", "case", "stage", "max|diff|", "rel", "verdict");

  for (auto it = man["cases"].begin(); it != man["cases"].end(); ++it) {
    const std::string name = it.key();
    const int T = it.value()["T"];
    const std::string d = fx + "/" + name;
    const int STRIDE = D.q_proj_out() + 2 * D.kv_proj_out();

    auto h_q  = load<uint16_t>(d + "/qkv_q.bf16", size_t(T) * D.q_proj_out());
    auto h_k  = load<uint16_t>(d + "/qkv_k.bf16", size_t(T) * D.kv_proj_out());
    auto h_v  = load<uint16_t>(d + "/qkv_v.bf16", size_t(T) * D.kv_proj_out());
    auto h_pos = load<int32_t>(d + "/positions.i32", T);
    auto r_cos = load<float>(d + "/cos.f32", D.rotary_dim);
    auto r_sin = load<float>(d + "/sin.f32", D.rotary_dim);
    auto r_qr  = load<uint16_t>(d + "/q_roped.bf16", size_t(T) * NQ * HD);
    auto r_k8  = load<uint16_t>(d + "/k_fp8.bf16", size_t(T) * NKV * HD);
    auto r_v8  = load<uint16_t>(d + "/v_fp8.bf16", size_t(T) * NKV * HD);
    auto r_att = load<uint16_t>(d + "/attn_fp8kv.bf16", size_t(T) * NQ * HD);
    auto r_gt  = load<uint16_t>(d + "/gate.bf16", size_t(T) * NQ * HD);
    auto r_gd  = load<uint16_t>(d + "/gated.bf16", size_t(T) * NQ * HD);

    // interleave the three projections into the fused layout the model produces
    std::vector<uint16_t> fused(size_t(T) * STRIDE);
    for (int t = 0; t < T; ++t) {
      memcpy(&fused[size_t(t)*STRIDE], &h_q[size_t(t)*D.q_proj_out()], D.q_proj_out()*2);
      memcpy(&fused[size_t(t)*STRIDE + D.q_proj_out()], &h_k[size_t(t)*D.kv_proj_out()], D.kv_proj_out()*2);
      memcpy(&fused[size_t(t)*STRIDE + D.q_proj_out() + D.kv_proj_out()],
             &h_v[size_t(t)*D.kv_proj_out()], D.kv_proj_out()*2);
    }

    __nv_bfloat16 *d_in, *d_qo, *d_gate, *d_out;
    uint8_t *d_kc, *d_vc;
    float *d_cos, *d_sin;
    int32_t* d_pos;
    CK(cudaMalloc(&d_in, fused.size()*2)); CK(cudaMemcpy(d_in, fused.data(), fused.size()*2, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_pos, T*4)); CK(cudaMemcpy(d_pos, h_pos.data(), T*4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_qo, size_t(T)*NQ*HD*2));
    CK(cudaMalloc(&d_gate, size_t(T)*NQ*HD*2));
    CK(cudaMalloc(&d_out, size_t(T)*NQ*HD*2));
    CK(cudaMalloc(&d_kc, size_t(T)*NKV*HD)); CK(cudaMalloc(&d_vc, size_t(T)*NKV*HD));
    CK(cudaMalloc(&d_cos, size_t(T)*HALF*4)); CK(cudaMalloc(&d_sin, size_t(T)*HALF*4));

    qwen::rope_tables(d_cos, d_sin, d_pos, d_pos, d_pos, T, D.rotary_dim, D.rope_theta);
    qwen::attn_prepare(d_qo, d_gate, d_kc, d_vc, d_in, d_qn, d_kn, d_cos, d_sin,
                       T, 0, T, D);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());

    auto fb = [&](__nv_bfloat16* p, size_t n){ std::vector<uint16_t> v(n);
      CK(cudaMemcpy(v.data(), p, n*2, cudaMemcpyDeviceToHost)); return v; };

    // cos/sin: the reference stores cat(freqs,freqs), so the first half is ours
    std::vector<float> gc(HALF), gs(HALF), wc(r_cos.begin(), r_cos.begin()+HALF),
                       ws(r_sin.begin(), r_sin.begin()+HALF);
    CK(cudaMemcpy(gc.data(), d_cos, HALF*4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gs.data(), d_sin, HALF*4, cudaMemcpyDeviceToHost));

    struct Row { const char* s; Stat st; double tol; };
    std::vector<Row> rows;
    // The reference casts cos/sin to bf16 before rotating, so BOTH sides carry 8
    // mantissa bits and the residual is which of two adjacent bf16 values a
    // float32 angle rounds to. At position 131071 the float32 angle resolution
    // is ~0.008 rad, so this depends on the exact bits of torch's inv_freq. One
    // bf16 ulp (2^-8 near 1.0) is a tie, not an error -- and it is 200x tighter
    // than the __powf bug this gate originally caught, which put cos 8e-3 out
    // and the resulting q 90% out.
    rows.push_back({"rope cos", cmp_f(gc, wc), 4.1e-3});
    rows.push_back({"rope sin", cmp_f(gs, ws), 4.1e-3});
    rows.push_back({"q norm+rope", cmp_b(fb(d_qo, size_t(T)*NQ*HD), r_qr), 8e-3});
    rows.push_back({"gate split", cmp_b(fb(d_gate, size_t(T)*NQ*HD), r_gt), 8e-3});

    // the fp8 cache must reproduce the reference's float8_e4m3fn round trip
    {
      std::vector<uint8_t> kc(size_t(T)*NKV*HD), vc(kc.size());
      CK(cudaMemcpy(kc.data(), d_kc, kc.size(), cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(vc.data(), d_vc, vc.size(), cudaMemcpyDeviceToHost));
      // decode e4m3 on the host the same way the kernel does
      auto deq = [](uint8_t b){ uint32_t x=b,e=(x>>3)&0xF,m=x&0x7;
        uint32_t bits=((e+120u)<<23)|(m<<20); float vn; memcpy(&vn,&bits,4);
        float v = e==0 ? float(m)/512.f : vn; return (x&0x80u)? -v : v; };
      std::vector<uint16_t> gk(kc.size()), gv(vc.size());
      auto tob = [](float f){ uint32_t u; memcpy(&u,&f,4);
        uint32_t r=(u>>16)&1u,b=u+0x7fffu+r; return uint16_t(b>>16); };
      for (size_t i = 0; i < kc.size(); ++i) { gk[i]=tob(deq(kc[i])); gv[i]=tob(deq(vc[i])); }
      // One e4m3 ulp is 2^-3 relative for a normal. Our roped K differs from
      // the reference's by bf16 noise, so values near a quantisation boundary
      // land one code either side; that is a tie, not an error. Exact-match
      // fraction is reported so a real regression still shows.
      size_t exk = 0, exv = 0;
      for (size_t i = 0; i < gk.size(); ++i) { exk += (gk[i]==r_k8[i]); exv += (gv[i]==r_v8[i]); }
      printf("%-12s %-16s  exact %.2f%% / %.2f%% (K/V bytes)\n", name.c_str(),
             "fp8 agreement", 100.0*exk/gk.size(), 100.0*exv/gv.size());
      rows.push_back({"K fp8 cache", cmp_b(gk, r_k8), 0.13});
      rows.push_back({"V fp8 cache", cmp_b(gv, r_v8), 0.13});
    }

    // prefill attention over the whole span
    {
      __nv_bfloat16* kvs; float* ss;
      CK(cudaMalloc(&kvs, size_t(2)*2048*NKV*HD*2));
      CK(cudaMalloc(&ss, qwen::attn_prefill_scratch_bytes(D, 2048, 128)));
      qwen::attn_prefill(d_out, d_qo, d_kc, d_vc, T, T, 0, T, D, kvs, ss, cb);
      CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
      rows.push_back({"prefill attn", cmp_b(fb(d_out, size_t(T)*NQ*HD), r_att), 3e-2});
      cudaFree(kvs); cudaFree(ss);
    }

    // decode attention: last query against the whole cached span
    {
      const int splits = qwen::attn_decode_splits(T);
      float* ws; CK(cudaMalloc(&ws, qwen::attn_decode_workspace_bytes(D, splits)));
      __nv_bfloat16* dout; CK(cudaMalloc(&dout, size_t(NQ)*HD*2));
      qwen::attn_decode(dout, d_qo + size_t(T-1)*NQ*HD, d_kc, d_vc, T, T, D, ws, splits);
      CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
      std::vector<uint16_t> want(r_att.begin() + size_t(T-1)*NQ*HD, r_att.end());
      rows.push_back({"decode attn", cmp_b(fb(dout, size_t(NQ)*HD), want), 3e-2});
      cudaFree(ws); cudaFree(dout);
    }

    // output gate
    {
      CK(cudaMemcpy(d_out, r_att.data(), r_att.size()*2, cudaMemcpyHostToDevice));
      qwen::attn_output_gate(d_out, d_gate, int(r_att.size()));
      CK(cudaDeviceSynchronize());
      rows.push_back({"output gate", cmp_b(fb(d_out, r_att.size()), r_gd), 8e-3});
    }

    for (auto& r : rows) {
      const bool ok = r.st.rel <= r.tol;
      if (!ok) ++nfail;
      printf("%-12s %-16s %11.3e %11.2e %8s\n", name.c_str(), r.s,
             r.st.maxabs, r.st.rel, ok ? "ok" : "FAIL");
    }
    cudaFree(d_in); cudaFree(d_pos); cudaFree(d_qo); cudaFree(d_gate);
    cudaFree(d_out); cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_cos); cudaFree(d_sin);
  }
  printf("\ngate_attn: %zu failures -> %s\n", nfail, nfail ? "FAIL" : "PASS");
  cublasDestroy(cb);
  return nfail ? 1 : 0;
}
