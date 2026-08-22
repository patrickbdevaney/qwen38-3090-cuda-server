// End-to-end cost of a speculative verification block against a decode step.
//
// This is the number that decides whether speculation is worth doing at all:
// if verifying T tokens costs T decode steps, there is nothing to gain.
#include <cstdio>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "../src/model/model.h"

int main(int argc, char** argv) {
  const std::string md = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  qwen::Model m; qwen::LoadOptions o;
  o.max_ctx = 8192; o.max_batch = 4096; o.lm_head_bits = 8; o.verbose = false;
  o.kv_k = qwen::KvFmt::INT4; o.kv_v = qwen::KvFmt::INT4;
  // argv[2]: GGUF weights. The two runners take different code paths above T=1
  // -- AWQ switches to the tensor-core GEMM, GGUF stays on its multi-row GEMV --
  // and this is the bench that shows what that costs.
  if (argc > 2) { o.gguf = argv[2]; o.lm_head_bits = 0; o.embed_host = true; }
  qwen::model_load(m, md, o);
  printf("%s\n", o.gguf.empty() ? "AWQ INT4 g128" : o.gguf.c_str());
  qwen::model_graph_capture(m);

  // warm a 2K context
  std::vector<int32_t> warm(2048, 100);
  qwen::model_prefill(m, warm.data(), int(warm.size()), 0);
  cudaDeviceSynchronize();
  const int pos = int(warm.size());

  auto time_it = [&](int T) {
    std::vector<int32_t> ids(T, 100);
    for (int i = 0; i < 3; ++i) {
      if (T == 1) qwen::model_decode(m, 100, pos);
      else        qwen::model_prefill(m, ids.data(), T, pos);
    }
    cudaDeviceSynchronize();
    std::vector<double> ms;
    for (int i = 0; i < 10; ++i) {
      const auto a = std::chrono::steady_clock::now();
      if (T == 1) qwen::model_decode(m, 100, pos);
      else        qwen::model_prefill(m, ids.data(), T, pos);
      cudaDeviceSynchronize();
      const auto b = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
  };

  // T=1 through the eager path too. Without this row the T>=2 numbers conflate
  // "block verification is expensive" with "T=1 replays a CUDA graph and
  // everything else pays ~1750 kernel launches".
  auto time_eager1 = [&]() {
    std::vector<int32_t> ids(1, 100);
    for (int i = 0; i < 3; ++i) qwen::model_prefill(m, ids.data(), 1, pos);
    cudaDeviceSynchronize();
    std::vector<double> ms;
    for (int i = 0; i < 10; ++i) {
      const auto a = std::chrono::steady_clock::now();
      qwen::model_prefill(m, ids.data(), 1, pos);
      cudaDeviceSynchronize();
      const auto b = std::chrono::steady_clock::now();
      ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
  };
  const double d1 = time_it(1);
  const double de = time_eager1();
  qwen::dbg_profile_report("T=1 eager");
  printf("%6s %10s %10s %14s %16s\n", "T", "ms", "vs T=1", "tokens if all", "geom p=0.78");
  printf("%6d %10.2f %9.2fx %14s %16s\n", 1, d1, 1.0, "1", "1.00");
  printf("%6s %10.2f %9.2fx %14s %16s\n", "1e", de, de / d1, "1", "(eager)");
  for (int T : {2, 4, 8, 16}) {
    const double t = time_it(T);
    { char tag[32]; snprintf(tag, sizeof tag, "T=%d", T); qwen::dbg_profile_report(tag); }
    // Expected committed tokens for a block of T (T-1 drafted plus the anchor's
    // free token), under a geometric per-position acceptance fitted to the
    // DFlash2 card's 4.39 mean at block 8.
    const double p = 0.78;
    double e = 1.0, q = 1.0;
    for (int i = 1; i < T; ++i) { q *= p; e += q; }
    printf("%6d %10.2f %9.2fx %14d %13.2f -> %.2fx\n", T, t, t / d1, T, e, e / (t / d1));
  }
  printf("\ndecode step %.2f ms; the drafter must fit in the gap between\n"
         "the block cost and the tokens it buys.\n", d1);
  return 0;
}
