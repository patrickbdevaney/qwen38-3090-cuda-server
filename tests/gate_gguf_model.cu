// Where does the GGUF path diverge from the AWQ path?
//
// Both checkpoints are the same model at different quantisations, so their
// residual streams should track each other to within quantisation noise --
// order 1e-2 relative, growing slowly with depth. A layer where the relative
// difference jumps to order 1 is a wiring bug (wrong tensor, wrong orientation,
// wrong split), not quantisation, and this prints the first one.
//
// Run both models in turn rather than together: two 12 GiB bodies do not fit.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../src/model/model.h"
#include "../src/tokenizer/bpe.h"

static float b2f(uint16_t h){ uint32_t u=uint32_t(h)<<16; float f; memcpy(&f,&u,4); return f; }

// Snapshot the SMALL bf16 tensors of one layer. These are the ones both loaders
// should agree on almost exactly: GGUF stores them as F32 and AWQ as bf16, so
// the only difference is a rounding. A large difference here is a mis-mapped
// tensor name, which is the most likely wiring mistake and the cheapest to
// check -- far cheaper than reasoning backwards from a wrong residual.
struct Smalls {
  std::vector<uint16_t> input_ln, post_ln, a, b, conv_w, A_log, dt_bias, gnorm;
};

static std::vector<uint16_t> grab(const __nv_bfloat16* d, int n) {
  std::vector<uint16_t> h(n);
  if (d) cudaMemcpy(h.data(), d, size_t(n) * 2, cudaMemcpyDeviceToHost);
  return h;
}

static Smalls smalls_of(const qwen::Model& m, int li) {
  const qwen::LayerWeights& L = m.layers[li];
  Smalls s;
  s.input_ln = grab(L.input_ln, m.shape.hidden_size);
  s.post_ln  = grab(L.post_ln,  m.shape.hidden_size);
  s.a        = grab(L.gdn_a, m.gdn.num_v_heads * m.shape.hidden_size);
  s.b        = grab(L.gdn_b, m.gdn.num_v_heads * m.shape.hidden_size);
  s.conv_w   = grab(L.gdn_conv_w, m.gdn.conv_dim() * m.gdn.conv_k);
  s.A_log    = grab(L.gdn_A_log, m.gdn.num_v_heads);
  s.dt_bias  = grab(L.gdn_dt_bias, m.gdn.num_v_heads);
  s.gnorm    = grab(L.gdn_norm, m.gdn.head_v);
  return s;
}

static void cmp(const char* name, const std::vector<uint16_t>& x,
                const std::vector<uint16_t>& y) {
  if (x.size() != y.size()) { printf("  %-10s SIZE %zu vs %zu\n", name, x.size(), y.size()); return; }
  double num = 0, den = 0; float mx = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    const double u = b2f(x[i]), v = b2f(y[i]);
    num += (u - v) * (u - v); den += u * u;
    mx = std::max(mx, float(std::fabs(u - v)));
  }
  const double rel = std::sqrt(num / std::max(den, 1e-30));
  printf("  %-10s n=%-8zu rel %10.3e  max|d| %10.3e  %s\n", name, x.size(), rel, mx,
         rel < 0.05 ? "ok" : "*** MISMATCH ***");
  if (rel >= 0.05) {
    printf("        awq :"); for (int i = 0; i < 6 && i < int(x.size()); ++i) printf(" %9.5f", b2f(x[i]));
    printf("\n        gguf:"); for (int i = 0; i < 6 && i < int(y.size()); ++i) printf(" %9.5f", b2f(y[i]));
    printf("\n");
  }
}

static std::vector<uint16_t> run(const std::string& md, const std::string& gguf,
                                 const std::vector<int32_t>& ids, int* NL, int* H,
                                 Smalls* sm = nullptr, int sm_layer = 0) {
  qwen::Model m;
  qwen::LoadOptions o;
  o.max_ctx = 2048; o.max_batch = 256; o.lm_head_bits = gguf.empty() ? 16 : 0;
  o.embed_host = true; o.verbose = false; o.gguf = gguf;
  qwen::model_load(m, md, o);
  *NL = m.shape.num_hidden_layers; *H = m.shape.hidden_size;
  cudaMalloc(&m.dbg_hidden, size_t(*NL + 1) * m.max_batch * *H * 2);
  if (sm) *sm = smalls_of(m, sm_layer);
  qwen::model_prefill(m, ids.data(), int(ids.size()), 0);
  cudaDeviceSynchronize();
  // Keep only the LAST position of each layer: that is where the answer forms
  // and it makes the comparison one vector per layer instead of T of them.
  std::vector<uint16_t> out(size_t(*NL + 1) * *H);
  const int last = int(ids.size()) - 1;
  for (int l = 0; l <= *NL; ++l)
    cudaMemcpy(out.data() + size_t(l) * *H,
               m.dbg_hidden + (size_t(l) * m.max_batch + last) * *H,
               size_t(*H) * 2, cudaMemcpyDeviceToHost);
  return out;
}

int main(int argc, char** argv) {
  const std::string md = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const std::string gg = argc > 2 ? argv[2]
      : "/home/patrickd/qwen38-weights/gguf/Qwen3.8-27B-UD-Q3_K_XL.gguf";

  qwen::Tokenizer tok;
  tok.load(md + "/tokenizer.json");
  std::vector<int32_t> ids = tok.encode("The capital of France is Paris. The capital of Germany is");
  printf("gate_gguf_model: %zu tokens\n", ids.size());

  int NL = 0, H = 0, NL2 = 0, H2 = 0;
  Smalls sa, sg;
  std::vector<uint16_t> a = run(md, "", ids, &NL, &H, &sa, 0);
  std::vector<uint16_t> g = run(md, gg, ids, &NL2, &H2, &sg, 0);

  printf("\nlayer 0 small tensors, AWQ vs GGUF:\n");
  cmp("input_ln", sa.input_ln, sg.input_ln);
  cmp("post_ln",  sa.post_ln,  sg.post_ln);
  cmp("gdn_a",    sa.a,        sg.a);
  cmp("gdn_b",    sa.b,        sg.b);
  cmp("conv_w",   sa.conv_w,   sg.conv_w);
  cmp("A_log",    sa.A_log,    sg.A_log);
  cmp("dt_bias",  sa.dt_bias,  sg.dt_bias);
  cmp("gdn_norm", sa.gnorm,    sg.gnorm);
  if (NL != NL2 || H != H2) { printf("shape mismatch\n"); return 2; }

  printf("\n%6s %12s %12s %12s\n", "layer", "rel diff", "|awq|", "|gguf|");
  int first_bad = -1;
  for (int l = 0; l <= NL; ++l) {
    double num = 0, den = 0, na = 0, ng = 0;
    for (int i = 0; i < H; ++i) {
      const double x = b2f(a[size_t(l) * H + i]), y = b2f(g[size_t(l) * H + i]);
      num += (x - y) * (x - y); den += x * x;
      na += x * x; ng += y * y;
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    if (l <= 8 || l % 8 == 0 || l == NL || (first_bad < 0 && rel > 0.35))
      printf("%6d %12.4f %12.2f %12.2f\n", l, rel, std::sqrt(na), std::sqrt(ng));
    if (first_bad < 0 && rel > 0.35) first_bad = l;
  }
  printf("\n");
  if (first_bad < 0) { printf("  RESULT: PASS (tracks within quantisation noise)\n"); return 0; }
  printf("  first layer over 0.35 relative: %d (%s)\n", first_bad,
         first_bad == 0 ? "embedding" : "layer body");
  printf("  RESULT: FAIL\n");
  return 1;
}
