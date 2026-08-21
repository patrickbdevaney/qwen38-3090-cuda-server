// Fused GGUF GEMV bandwidth, against the AWQ GEMV as the calibration.
//
// Decode is bandwidth bound, so the only thing that matters is bytes/second off
// DRAM. A smaller quant only pays if the kernel reading it keeps up.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "../src/gguf/gguf.h"
#include "../src/gguf/gemv.cuh"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA %s\n",cudaGetErrorString(e)); return 2; } } while(0)

int main(int argc, char** argv) {
  const std::string path = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/gguf/Qwen3.8-27B-UD-Q3_K_XL.gguf";
  qwen::GgufFile f;
  try { f.open(path); }
  catch (const std::exception& e) { printf("cannot open: %s\n", e.what()); return 2; }

  // The widest tensors are what decode actually spends its time on.
  std::vector<std::string> want = {
      "blk.1.ffn_up.weight", "blk.1.ffn_down.weight", "blk.1.ffn_gate.weight",
      "blk.0.attn_qkv.weight", "blk.3.attn_q.weight", "output.weight"};
  printf("%-26s %-8s %10s %10s %10s %10s\n", "tensor", "type", "MiB", "ms", "GB/s",
         "%% of 914");
  double tot_bytes = 0, tot_ms = 0;
  for (const auto& n : want) {
    const qwen::GgufTensor* t = f.find(n);
    if (!t) continue;
    const int in_f = int(t->row_len()), out_f = int(t->rows());
    if (in_f % 256) continue;
    const size_t nb = qwen::gguf_row_bytes(t->type, in_f) * size_t(out_f);

    void* d_w = nullptr; __nv_bfloat16 *d_x = nullptr, *d_y = nullptr;
    if (cudaMalloc(&d_w, nb) != cudaSuccess) { printf("  %-26s too big\n", n.c_str()); continue; }
    CK(cudaMemcpy(d_w, t->data, nb, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_x, size_t(in_f) * 2));
    CK(cudaMalloc(&d_y, size_t(out_f) * 2));
    CK(cudaMemset(d_x, 0, size_t(in_f) * 2));

    qwen::GgufWeight w;
    w.data = d_w; w.type = t->type; w.out_f = out_f; w.in_f = in_f;
    for (int i = 0; i < 3; ++i) qwen::gguf_gemv(d_y, w, d_x);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int IT = 20;
    cudaEventRecord(a);
    for (int i = 0; i < IT; ++i) qwen::gguf_gemv(d_y, w, d_x);
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= IT;
    const double gbs = double(nb) / (ms * 1e-3) / 1e9;
    printf("%-26s %-8s %10.1f %10.3f %10.1f %9.1f%%\n", n.c_str(),
           qwen::ggml_type_name(t->type), double(nb) / (1 << 20), ms, gbs,
           100.0 * gbs / 914.2);
    tot_bytes += double(nb); tot_ms += ms;
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_y);
  }
  if (tot_ms > 0)
    printf("\naggregate %.1f GB/s (%.1f%% of the 914.2 GB/s measured DRAM read)\n",
           tot_bytes / (tot_ms * 1e-3) / 1e9,
           100.0 * (tot_bytes / (tot_ms * 1e-3) / 1e9) / 914.2);
  printf("AWQ INT4 g128 reference: 769.8 GB/s aggregate = 84.2%%\n");
  return 0;
}
