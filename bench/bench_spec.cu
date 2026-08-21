// Speculative decoding throughput and the k-sweep.
//
// The per-position acceptance histogram is the diagnostic that says whether the
// block size is right: if the last slot is never accepted, the block is too
// long and the extra verification is pure cost.
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../src/model/model.h"
#include "../src/spec/spec.h"

int main(int argc, char** argv) {
  const std::string md = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const int NGEN = argc > 2 ? atoi(argv[2]) : 192;

  qwen::Model m; qwen::LoadOptions o;
  o.max_ctx = 8192; o.max_batch = 2048; o.lm_head_bits = 8; o.verbose = false;
  qwen::model_load(m, md, o);
  qwen::model_graph_capture(m);
  qwen::SpecState sp;
  qwen::spec_alloc(sp, m, 16);

  struct P { const char* name; std::vector<int32_t> ids; };
  const std::vector<P> prompts = {
    {"code",   {750, 84922, 1445, 982, 262, 421, 308, 366, 220, 17, 510, 286, 470, 308, 198, 262, 470}},
    {"repeat", {4340, 1657, 3039, 1558, 279, 3409, 330, 1944, 1, 4994, 304, 25, 1944, 1944, 1944, 1944}},
    {"prose",  {785, 6722, 315, 9625, 374}},
  };

  // baseline
  printf("%-8s %10s %12s %10s\n", "prompt", "AR tok/s", "AR ms/token", "tokens");
  std::vector<double> ar(prompts.size());
  for (size_t i = 0; i < prompts.size(); ++i) {
    cudaMemset(m.gdn_state, 0, size_t(m.shape.gdn_state_elems()) * 4);
    cudaMemset(m.gdn_conv, 0, size_t(m.shape.gdn_conv_state_elems()) * 4);
    const auto t0 = std::chrono::steady_clock::now();
    auto out = qwen::model_generate_greedy(m, prompts[i].ids, NGEN, m.shape.eos_token_id);
    cudaDeviceSynchronize();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ar[i] = out.size() / s;
    printf("%-8s %10.1f %12.2f %10zu\n", prompts[i].name, ar[i], s * 1000 / out.size(), out.size());
  }

  printf("\n%-8s %4s %10s %8s %12s %10s %s\n", "prompt", "k", "tok/s", "vs AR",
         "mean accept", "rounds", "per-position acceptance histogram");
  for (size_t i = 0; i < prompts.size(); ++i) {
    for (int k : {1, 2, 3, 5, 7, 11, 15}) {
      cudaMemset(m.gdn_state, 0, size_t(m.shape.gdn_state_elems()) * 4);
      cudaMemset(m.gdn_conv, 0, size_t(m.shape.gdn_conv_state_elems()) * 4);
      qwen::SpecStats st;
      const auto t0 = std::chrono::steady_clock::now();
      auto out = qwen::spec_generate(m, sp, prompts[i].ids, NGEN, k, st);
      cudaDeviceSynchronize();
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      const double tps = out.size() / s;
      printf("%-8s %4d %10.1f %7.2fx %12.2f %10llu  ", prompts[i].name, k, tps, tps / ar[i],
             st.mean_acceptance(), (unsigned long long)st.rounds);
      for (size_t j = 0; j < st.per_position.size() && int(j) <= k; ++j)
        printf("%llu ", (unsigned long long)st.per_position[j]);
      printf("\n");
    }
  }
  qwen::spec_free(sp);
  return 0;
}
