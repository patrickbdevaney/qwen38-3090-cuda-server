// GATE: the CUDA DFlash2 drafter against the OFFICIAL z-lab/dflash reference.
//
// The dump is produced by tools/dump_dflash_ref.py, which drives
// DFlash2DraftModel itself rather than my reading of it. Synthetic context
// hidden states are used deliberately: the drafter's numerics do not depend on
// where the target's residual stream came from, and feeding random taps means
// this gate needs no 27B forward.
//
// Every comparison flags non-finite values explicitly. A previous gate in this
// repo passed an all-NaN tensor because std::max(x, NaN) returns x, so the
// measured difference was zero. A gate that cannot fail is worse than no gate.
//
// ON THE TENSOR TOLERANCE. The committed fixture is the reference run in bf16,
// which is what the checkpoint is. Running the same reference in fp32 and
// comparing three ways shows the residual is the REFERENCE's rounding, not this
// implementation's error -- at every stage the CUDA output sits closer to fp32
// than the bf16 reference does:
//
//                     vs bf16 ref     vs fp32 ref
//     l0_attn           9.45e-3         6.64e-3
//     l0_out            2.33e-2         1.66e-2
//     draft_hidden      2.80e-2         1.93e-2
//
// So 6e-2 is the measured bf16 floor for 5 layers with two dynamic convolutions
// each, not a number picked to make this pass. The HARD gate is the proposed
// path: the drafter's actual output is 7 token ids, and those must match the
// reference exactly. Candidate-set membership is reported but not gated -- the
// two references disagree with each other at 5 of 112 slots, because the 16th
// ranked logit is a near tie, and a candidate that never becomes the path
// cannot change what gets proposed. Every drafted token is verified by the
// target regardless, so a drafter disagreement costs throughput, never
// correctness.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../third_party/json.hpp"
#include "../src/draft/dflash.h"
#include "../src/loader/safetensors.h"

using json = nlohmann::json;

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 2; } } while(0)

static float b2f(uint16_t h) { uint32_t u = uint32_t(h) << 16; float f; memcpy(&f, &u, 4); return f; }

template <class T>
static std::vector<T> read_bin(const std::string& p, size_t n) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { printf("missing fixture %s\n", p.c_str()); exit(2); }
  std::vector<T> v(n);
  f.read(reinterpret_cast<char*>(v.data()), n * sizeof(T));
  if (size_t(f.gcount()) != n * sizeof(T)) { printf("short read %s\n", p.c_str()); exit(2); }
  return v;
}

struct Cmp { double maxabs = 0, maxrel = 0, rms = 0; long nonfinite = 0; };

static Cmp compare(const std::vector<uint16_t>& got, const std::vector<uint16_t>& want) {
  Cmp c;
  double se = 0, sw = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    const float a = b2f(got[i]), b = b2f(want[i]);
    if (!std::isfinite(a) || !std::isfinite(b)) { ++c.nonfinite; continue; }
    const double d = std::fabs(double(a) - double(b));
    c.maxabs = std::max(c.maxabs, d);
    se += d * d; sw += double(b) * double(b);
  }
  c.rms = want.size() ? std::sqrt(se / double(want.size())) : 0;
  c.maxrel = sw > 0 ? std::sqrt(se / sw) : 0;
  return c;
}

static bool report(const char* name, const Cmp& c, double tol) {
  const bool ok = c.nonfinite == 0 && c.maxrel <= tol;
  printf("  %-16s max|d| %10.3e  rel %10.3e  nonfinite %5ld   %s\n",
         name, c.maxabs, c.maxrel, c.nonfinite, ok ? "OK" : "FAIL");
  return ok;
}

int main(int argc, char** argv) {
  const std::string ddir = argc > 1 ? argv[1] : "/home/patrickd/qwen38-weights/Qwen3.8-27B-DFlash2";
  const std::string fx = argc > 2 ? argv[2] : "tests/fixtures/dflash";
  const std::string bf16 = argc > 3 ? argv[3] : "";

  json man; { std::ifstream f(fx + "/manifest.json"); if (!f) { printf("no manifest in %s\n", fx.c_str()); return 2; } f >> man; }
  const int H = man["hidden"], BS = man["block_size"], NT = man["num_taps"];
  const int T = man["ctx"], K = man["top_k"], R = man["rank"];

  qwen::DraftModel d;
  qwen::DraftLoadOptions o;
  o.quantize = false;
  o.debug = true;
  o.ctx_chunk = std::max(T, 64);
  qwen::draft_load(d, ddir, o);

  // shape agreement with the reference's own manifest
  bool ok = true;
  auto eq = [&](const char* n, int a, int b) {
    if (a != b) { printf("  shape %s: %d != %d (reference)\n", n, a, b); ok = false; }
  };
  eq("hidden", d.sh.hidden, H);
  eq("block_size", d.sh.block_size, BS);
  eq("num_taps", d.sh.n_taps, NT);
  eq("top_k", d.sh.top_k, K);
  eq("rank", d.sh.rank, R);
  eq("mask_token_id", d.sh.mask_token_id, man["mask_token_id"].get<int>());
  eq("sliding_window", d.sh.sliding_window, man["sliding_window"].get<int>());
  if (!ok) return 1;

  auto th = read_bin<uint16_t>(fx + "/target_hidden.bf16", size_t(T) * NT * H);
  auto ne = read_bin<uint16_t>(fx + "/noise_embedding.bf16", size_t(BS) * H);
  auto want_hidden = read_bin<uint16_t>(fx + "/hidden_out.bf16", size_t(BS) * H);
  auto want_draft = read_bin<uint16_t>(fx + "/draft_hidden.bf16", size_t(BS - 1) * H);
  auto want_hp = read_bin<uint16_t>(fx + "/hidden_proj.bf16", size_t(BS - 1) * R);
  auto want_path = read_bin<int32_t>(fx + "/path.i32", BS - 1);
  auto want_cand = read_bin<int32_t>(fx + "/candidates.i32", size_t(BS - 1) * K);

  __nv_bfloat16* d_th = nullptr;
  CK(cudaMalloc(&d_th, th.size() * 2));
  CK(cudaMemcpy(d_th, th.data(), th.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.noise, ne.data(), ne.size() * 2, cudaMemcpyHostToDevice));

  const __nv_bfloat16* out = qwen::draft_forward(d, d_th, T, 0, T, /*use_cache=*/false);
  CK(cudaDeviceSynchronize());

  printf("\ngate_dflash (ctx %d, block %d)\n", T, BS);
  {
    struct { const char* file; int idx; int rows; } taps[] = {
      {"ctx_norm",         qwen::DraftModel::DBG_CTX_NORM,   T},
      {"l0_ln",            qwen::DraftModel::DBG_L0_LN,      BS},
      {"l0_conv_prepare",  qwen::DraftModel::DBG_L0_CONV,    BS},
      {"l0_attn",          qwen::DraftModel::DBG_L0_ATTN,    BS},
      {"l0_post_ln",       qwen::DraftModel::DBG_L0_POST_LN, BS},
      {"l0_out",           qwen::DraftModel::DBG_L0_OUT,     BS},
    };
    for (auto& t2 : taps) {
      const std::string path = fx + "/stage_" + t2.file + ".bf16";
      std::ifstream probe(path, std::ios::binary);
      if (!probe) continue;
      probe.close();
      auto wantv = read_bin<uint16_t>(path, size_t(t2.rows) * H);
      std::vector<uint16_t> gotv(wantv.size());
      CK(cudaMemcpy(gotv.data(), d.dbg[t2.idx], gotv.size() * 2, cudaMemcpyDeviceToHost));
      report(t2.file, compare(gotv, wantv), 6e-2);
    }
  }
  std::vector<uint16_t> got(size_t(BS) * H);
  CK(cudaMemcpy(got.data(), out, got.size() * 2, cudaMemcpyDeviceToHost));
  ok &= report("hidden_out", compare(got, want_hidden), 6e-2);

  std::vector<uint16_t> got_draft(got.begin() + H, got.end());
  ok &= report("draft_hidden", compare(got_draft, want_draft), 6e-2);

  if (bf16.empty()) {
    printf("  (no BF16 target dir given: skipping the logit, candidate and path\n"
           "   checks, which need the target's lm_head. Pass it as argv[3].)\n");
    printf("  RESULT: %s (partial)\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  }

  // The drafter has no head of its own; logits come from the TARGET's.
  qwen::SafeTensors st;
  st.open_dir(bf16);
  const qwen::TensorView& lh = st.get("lm_head.weight");
  const int V = int(lh.shape[0]);
  __nv_bfloat16* d_lh = nullptr;
  CK(cudaMalloc(&d_lh, lh.nbytes));
  CK(cudaMemcpy(d_lh, lh.data, lh.nbytes, cudaMemcpyHostToDevice));
  st.close();

  const int NP = BS - 1;
  __nv_bfloat16* d_logits = nullptr;
  CK(cudaMalloc(&d_logits, size_t(NP) * V * 2));
  cublasHandle_t hb; cublasCreate(&hb);
  const float one = 1.f, zero = 0.f;
  cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, V, NP, H, &one, d_lh, CUDA_R_16BF, H,
               out + size_t(1) * H, CUDA_R_16BF, H, &zero, d_logits, CUDA_R_16BF, V,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  CK(cudaDeviceSynchronize());

  // the anchor for the Markov chain is block_ids[0], the last committed token
  auto bids = read_bin<int32_t>(fx + "/block_ids.i32", BS);
  qwen::draft_select(d, out + size_t(1) * H, d_logits, NP, bids[0]);
  CK(cudaDeviceSynchronize());

  std::vector<uint16_t> got_hp(size_t(NP) * R);
  CK(cudaMemcpy(got_hp.data(), d.hp, got_hp.size() * 2, cudaMemcpyDeviceToHost));
  ok &= report("hidden_proj", compare(got_hp, want_hp), 6e-2);

  std::vector<int32_t> got_cand(size_t(NP) * K), got_path(NP);
  CK(cudaMemcpy(got_cand.data(), d.cand, got_cand.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(got_path.data(), d.path, got_path.size() * 4, cudaMemcpyDeviceToHost));

  // topk with sorted=False gives no order guarantee, so compare the SETS.
  int cand_match = 0;
  for (int p = 0; p < NP; ++p) {
    std::vector<int32_t> a(got_cand.begin() + size_t(p) * K, got_cand.begin() + size_t(p + 1) * K);
    std::vector<int32_t> b(want_cand.begin() + size_t(p) * K, want_cand.begin() + size_t(p + 1) * K);
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    for (int i = 0; i < K; ++i) if (std::binary_search(b.begin(), b.end(), a[i])) ++cand_match;
  }
  printf("  %-16s %d / %d candidates agree\n", "candidates", cand_match, NP * K);

  int path_match = 0;
  for (int p = 0; p < NP; ++p) if (got_path[p] == want_path[p]) ++path_match;
  printf("  %-16s %d / %d   ours [", "path", path_match, NP);
  for (int p = 0; p < NP; ++p) printf("%s%d", p ? "," : "", got_path[p]);
  printf("]  ref [");
  for (int p = 0; p < NP; ++p) printf("%s%d", p ? "," : "", want_path[p]);
  printf("]\n");

  ok &= (path_match == NP);
  printf("  RESULT: %s\n", ok ? "PASS" : "FAIL");
  cublasDestroy(hb);
  qwen::draft_free(d);
  return ok ? 0 : 1;
}
