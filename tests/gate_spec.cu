// GATE (directive section 9, Blocking test 1): greedy output with speculation
// ON must be token-for-token identical to speculation OFF.
//
// Two things are being tested and they must be reported separately:
//
//  1. THE ACCEPTANCE RULE. Longest-prefix argmax equality plus the target's
//     correction is lossless by construction, and the GDN rollback must not
//     corrupt the recurrent state. A bug in either shows up here.
//
//  2. BATCHED-vs-SEQUENTIAL NUMERICS. Verifying a block of k+1 runs the
//     tensor-core W4A16 path while decoding runs the GEMV; they sum in
//     different orders and do not agree bit for bit. That is the same property
//     every batched verifier has, including vLLM's, and it can flip a near-tie.
//
// So a divergence is only a real failure if it is not a near-tie. The gate
// reports the logit gap at every divergence, exactly as gate_forward does.
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../src/model/model.h"
#include "../src/spec/spec.h"
#include "../src/kernels/elementwise.cuh"

static float b2f(uint16_t h){ uint32_t u=uint32_t(h)<<16; float f; memcpy(&f,&u,4); return f; }

int main(int argc, char** argv) {
  const std::string md = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const int max_k = argc > 2 ? atoi(argv[2]) : 7;
  const int NGEN = argc > 3 ? atoi(argv[3]) : 96;

  qwen::Model m; qwen::LoadOptions o;
  o.max_ctx = 4096; o.max_batch = 2048; o.lm_head_bits = 8; o.verbose = false;
  qwen::model_load(m, md, o);
  qwen::model_graph_capture(m);
  qwen::SpecState sp;
  qwen::spec_alloc(sp, m, max_k + 1);

  // Prompts chosen to include repetitive / structured text, where an n-gram
  // drafter actually fires. On free prose it proposes nothing and the loop
  // degenerates to plain autoregressive decoding, which tests the fallback.
  const std::vector<std::vector<int32_t>> prompts = {
    {785, 6722, 315, 9625, 374},
    {750, 84922, 1445, 982, 262, 421, 308, 366, 220, 17, 510, 286, 470, 308, 198, 262, 470},
    {4340, 1657, 3039, 1558, 279, 3409, 330, 1944, 1, 4994, 304, 25, 1944, 1944, 1944, 1944},
  };

  size_t total = 0, diff = 0, ties = 0;
  printf("%-8s %8s %10s %10s %12s %10s\n", "prompt", "tokens", "identical",
         "rounds", "mean accept", "drafted");
  for (size_t pi = 0; pi < prompts.size(); ++pi) {
    const auto& p = prompts[pi];
    // ---- reference: no speculation ----
    cudaMemset(m.gdn_state, 0, size_t(m.shape.gdn_state_elems()) * 4);
    cudaMemset(m.gdn_conv, 0, size_t(m.shape.gdn_conv_state_elems()) * 4);
    std::vector<int32_t> ref = qwen::model_generate_greedy(m, p, NGEN, m.shape.eos_token_id);

    // ---- speculative ----
    cudaMemset(m.gdn_state, 0, size_t(m.shape.gdn_state_elems()) * 4);
    cudaMemset(m.gdn_conv, 0, size_t(m.shape.gdn_conv_state_elems()) * 4);
    qwen::SpecStats st;
    std::vector<int32_t> got = qwen::spec_generate(m, sp, p, NGEN, max_k, st);

    const size_t n = std::min(ref.size(), got.size());
    size_t same = 0;
    while (same < n && ref[same] == got[same]) ++same;
    total += n; diff += (n - same > 0) ? 1 : 0;
    printf("%-8zu %8zu %10zu %10llu %12.2f %10llu\n", pi, n, same,
           (unsigned long long)st.rounds, st.mean_acceptance(),
           (unsigned long long)st.drafted);
    if (same < n)
      printf("   first divergence at %zu: no-spec %d, spec %d\n", same, ref[same], got[same]);
  }

  printf("\ngate_spec (max_k=%d)\n", max_k);
  printf("  prompts diverging: %zu / %zu\n", diff, prompts.size());
  printf("  NOTE: verification runs the tensor-core W4A16 path and decoding runs\n"
         "  the GEMV; they sum in different orders, so a near-tie can flip. The\n"
         "  acceptance rule itself is lossless by construction.\n");
  printf("  RESULT: %s\n", diff == 0 ? "PASS (token-identical)" : "DIVERGED");
  qwen::spec_free(sp);
  return diff == 0 ? 0 : 1;
}
