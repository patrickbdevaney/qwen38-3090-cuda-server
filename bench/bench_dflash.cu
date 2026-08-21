// End-to-end DFlash2 speculative decode against plain autoregressive decode.
//
// Both paths are greedy and the acceptance rule is lossless, so the token
// streams must agree; this reports that alongside the speed, because a
// speculative decoder that is fast and wrong is worth nothing.
#include <cstdio>
#include <chrono>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "../src/spec/spec.h"

int main(int argc, char** argv) {
  const std::string md = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const std::string dd = argc > 2 ? argv[2]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-DFlash2";
  const int NGEN = argc > 3 ? atoi(argv[3]) : 192;
  const bool quant = argc > 4 ? atoi(argv[4]) != 0 : true;

  qwen::Model m; qwen::LoadOptions o;
  o.max_ctx = 4096; o.max_batch = 512; o.lm_head_bits = 8; o.verbose = false;
  qwen::model_load(m, md, o);

  qwen::DraftModel d; qwen::DraftLoadOptions po;
  po.ctx_chunk = 512;
  po.quantize = quant;
  qwen::draft_load(d, dd, po);

  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  printf("VRAM after target + drafter: %.2f GiB free of %.2f\n",
         double(free_b) / (1 << 30), double(total_b) / (1 << 30));

  struct P { const char* name; std::vector<int32_t> ids; };
  std::vector<P> prompts = {
    {"p0", {5501, 1044, 2000, 8, 19}},
    {"p1", {3923, 374, 279, 6864, 315, 9822, 30, 22559, 304, 832, 11914, 13}},
    {"p2", {9906, 11, 1268, 656, 358, 3460, 264, 1160, 304, 13325, 30}},
  };

  qwen::SpecState st;
  qwen::spec_alloc(st, m, d.sh.block_size);

  printf("\n%-6s %9s %9s %9s %8s %9s %9s %s\n", "prompt", "AR tok/s", "spec t/s",
         "speedup", "rounds", "mean acc", "same?", "acceptance histogram");
  double ar_sum = 0, sp_sum = 0;
  for (auto& p : prompts) {
    cudaMemset(m.gdn_state, 0, size_t(m.shape.gdn_state_elems()) * 4);
    cudaMemset(m.gdn_conv, 0, size_t(m.shape.gdn_conv_state_elems()) * 4);
    auto t0 = std::chrono::steady_clock::now();
    auto ar = qwen::model_generate_greedy(m, p.ids, NGEN, m.shape.eos_token_id);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    const double ar_s = std::chrono::duration<double>(t1 - t0).count();

    cudaMemset(m.gdn_state, 0, size_t(m.shape.gdn_state_elems()) * 4);
    cudaMemset(m.gdn_conv, 0, size_t(m.shape.gdn_conv_state_elems()) * 4);
    qwen::SpecStats stats;
    t0 = std::chrono::steady_clock::now();
    auto sp = qwen::spec_generate_dflash(m, st, d, p.ids, NGEN, stats);
    cudaDeviceSynchronize();
    t1 = std::chrono::steady_clock::now();
    const double sp_s = std::chrono::duration<double>(t1 - t0).count();

    size_t same = 0;
    while (same < ar.size() && same < sp.size() && ar[same] == sp[same]) ++same;
    const double artps = ar.size() / ar_s, sptps = sp.size() / sp_s;
    ar_sum += artps; sp_sum += sptps;
    printf("%-6s %9.1f %9.1f %8.2fx %8llu %9.2f %4zu/%zu  ", p.name, artps, sptps,
           sptps / artps, (unsigned long long)stats.rounds, stats.mean_acceptance(),
           same, ar.size());
    for (size_t i = 0; i < stats.per_position.size(); ++i)
      printf("%llu ", (unsigned long long)stats.per_position[i]);
    printf("\n");
  }
  printf("\nmean AR %.1f tok/s, mean spec %.1f tok/s, %.2fx\n",
         ar_sum / prompts.size(), sp_sum / prompts.size(), sp_sum / ar_sum);
  return 0;
}
