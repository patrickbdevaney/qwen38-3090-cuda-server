// End-to-end decode throughput: G1 (4K context) and G2 (64K vs 4K).
//
// This is the product's headline number. Per the directive: 3 warmup runs,
// 10 measured, median and p95.
#include <cstdio>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "../src/model/model.h"
#include "../src/kernels/elementwise.cuh"

int main(int argc, char** argv) {
  const std::string md = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  const int max_ctx = argc > 2 ? atoi(argv[2]) : 8192;
  const int quant_lm = argc > 3 ? atoi(argv[3]) : 1;

  qwen::Model m; qwen::LoadOptions o;
  o.max_ctx = max_ctx; o.max_batch = 4096;
  o.lm_head_bits = quant_lm; o.quantize_embed = true;
  qwen::model_load(m, md, o);
  printf("\nlm_head %d-bit, max_ctx %d\n", o.lm_head_bits, max_ctx);
  const bool use_graph = (argc > 4) ? atoi(argv[4]) != 0 : true;
  if (use_graph) qwen::model_graph_capture(m);
  m.use_graph = use_graph;

  int32_t* d_id = m.argmax_scratch + 512;
  printf("\n%8s %12s %12s %12s %10s\n", "ctx", "median ms", "p95 ms", "tok/s", "vs 4K");
  double base = 0;
  for (int ctx : {4096, 16384, 32768, 65536, 131072}) {
    if (ctx > max_ctx) break;
    // warm the cache to `ctx` with a chunked prefill
    cudaMemset(m.gdn_state, 0, size_t(m.shape.gdn_state_elems()) * 4);
    cudaMemset(m.gdn_conv, 0, size_t(m.shape.gdn_conv_state_elems()) * 4);
    std::vector<int32_t> ids(std::min(ctx, 4096), 100);
    int pos = 0;
    while (pos < ctx - 1) {
      const int n = std::min<int>(ids.size(), ctx - 1 - pos);
      qwen::model_prefill(m, ids.data(), n, pos);
      pos += n;
    }
    cudaDeviceSynchronize();

    std::vector<double> ms;
    for (int i = 0; i < 13; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      qwen::model_decode(m, 100, pos);
      qwen::argmax(d_id, m.logits, m.shape.vocab_size, m.argmax_scratch);
      int32_t tok; cudaMemcpy(&tok, d_id, 4, cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();
      const auto t1 = std::chrono::steady_clock::now();
      if (i >= 3) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2], p95 = ms[size_t(ms.size() * 0.95) - 1];
    if (!base) base = 1000.0 / med;
    printf("%8d %12.2f %12.2f %12.1f %9.0f%%\n", ctx, med, p95, 1000.0 / med,
           100.0 * (1000.0 / med) / base);
  }
  {
    // G7 is a PEAK number, not a post-load one: CUDA graph instantiation and
    // cuBLAS workspaces land after the budget print.
    size_t fb = 0, tb = 0;
    cudaMemGetInfo(&fb, &tb);
    printf("\npeak VRAM in use: %.0f MiB of %.0f (%.2f GB)   [G7 bar 22.5 GB at 128K]\n",
           double(tb - fb) / (1 << 20), double(tb) / (1 << 20),
           double(tb - fb) / 1e9);
  }
  printf("\nG1 gate: 45 tok/s min at 4K (revised up from 35; llama.cpp measures 38.41)\n");
  printf("G2 gate: 64K decode >= 85%% of 4K\n");
  return 0;
}
