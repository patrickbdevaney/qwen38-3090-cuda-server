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
  // One representative tensor per type PRESENT IN THE FILE, chosen automatically.
  // A hardcoded list left seven of the fourteen types unsampled, so the
  // composition-weighted number below covered 84% of the model and quietly
  // assumed the rest behaved like the average of what was measured. Picking the
  // largest tensor of each type that still fits comfortably makes the coverage
  // ~100% and the timing less sensitive to launch overhead.
  std::vector<std::string> want;
  {
    std::vector<std::pair<qwen::GgmlType, std::pair<double, std::string>>> best;
    for (const auto& kv : f.all()) {
      const qwen::GgufTensor& t = kv.second;
      if (int(t.row_len()) % 256) continue;
      if (!qwen::gguf_gemv_supported(t.type)) continue;
      const double nb = double(qwen::gguf_row_bytes(t.type, int(t.row_len()))) * double(t.rows());
      if (nb > 900.0 * (1 << 20)) continue;          // must fit alongside x and y
      bool found = false;
      for (auto& e : best)
        if (e.first == t.type) { found = true; if (nb > e.second.first) e.second = {nb, kv.first}; break; }
      if (!found) best.emplace_back(t.type, std::make_pair(nb, kv.first));
    }
    // Stable order so two runs of the bench print the same rows.
    std::sort(best.begin(), best.end(),
              [](const auto& a, const auto& b) { return int(a.first) < int(b.first); });
    for (const auto& e : best) want.push_back(e.second.second);
  }
  printf("%-26s %-8s %10s %10s %10s %10s\n", "tensor", "type", "MiB", "ms", "GB/s",
         "%% of 914");
  double tot_bytes = 0, tot_ms = 0;
  // Per-type rate, so the composition-weighted projection below can use it.
  std::vector<std::pair<qwen::GgmlType, double>> rate;
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
    rate.emplace_back(t->type, gbs);
    cudaFree(d_w); cudaFree(d_x); cudaFree(d_y);
  }
  // Does speculation compress the gap? At M rows the weight stream is read ONCE
  // and the dequantisation is amortised across M outputs, so an ALU-bound kernel
  // should improve almost linearly in M while a bandwidth-bound one stays flat.
  {
    const qwen::GgufTensor* t = f.find("blk.1.ffn_up.weight");
    if (t && t->row_len() % 256 == 0) {
      const int in_f = int(t->row_len()), out_f = int(t->rows());
      const size_t nb = qwen::gguf_row_bytes(t->type, in_f) * size_t(out_f);
      void* d_w = nullptr; __nv_bfloat16 *d_x = nullptr, *d_y = nullptr;
      CK(cudaMalloc(&d_w, nb));
      CK(cudaMemcpy(d_w, t->data, nb, cudaMemcpyHostToDevice));
      CK(cudaMalloc(&d_x, size_t(in_f) * 8 * 2));
      CK(cudaMalloc(&d_y, size_t(out_f) * 8 * 2));
      CK(cudaMemset(d_x, 0, size_t(in_f) * 8 * 2));
      qwen::GgufWeight w;
      w.data = d_w; w.type = t->type; w.out_f = out_f; w.in_f = in_f;
      printf("\nM sweep on %s (%s) -- speculation reads the weights once for M rows\n",
             t->name.c_str(), qwen::ggml_type_name(t->type));
      printf("%6s %10s %12s %14s\n", "M", "ms", "GB/s", "per-token ms");
      for (int M : {1, 2, 4, 8}) {
        for (int i = 0; i < 3; ++i) qwen::gguf_gemm_small(d_y, w, d_x, M);
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        const int IT = 20;
        cudaEventRecord(a);
        for (int i = 0; i < IT; ++i) qwen::gguf_gemm_small(d_y, w, d_x, M);
        cudaEventRecord(b);
        CK(cudaEventSynchronize(b));
        float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= IT;
        printf("%6d %10.3f %12.1f %14.4f\n", M, ms,
               double(nb) / (ms * 1e-3) / 1e9, ms / M);
      }
      cudaFree(d_w); cudaFree(d_x); cudaFree(d_y);
    }
  }

  if (tot_ms > 0)
    printf("\nsampled aggregate %.1f GB/s (%.1f%% of the 914.2 GB/s measured DRAM read)\n",
           tot_bytes / (tot_ms * 1e-3) / 1e9,
           100.0 * (tot_bytes / (tot_ms * 1e-3) / 1e9) / 914.2);

  // THE NUMBER THAT PREDICTS DECODE.
  //
  // The line above is one tensor per type, and output.weight alone is 834 of
  // the ~995 MiB sampled -- so it is very nearly a measurement of Q5_K, which
  // is 9% of this model. Weighting each type by the bytes it actually occupies
  // in the file gives the rate a decode step would see. On UD-Q3_K_XL that
  // composition is 38% IQ4_XS and 27% IQ3_S; Q3_K and Q5_K together are 16%.
  {
    std::vector<std::pair<qwen::GgmlType, double>> bytes_of;   // type -> bytes
    double model_bytes = 0;
    for (const auto& kv : f.all()) {
      const qwen::GgufTensor& t = kv.second;
      const int in_f = int(t.row_len());
      if (in_f % 256) continue;
      const double nb = double(qwen::gguf_row_bytes(t.type, in_f)) * double(t.rows());
      bool found = false;
      for (auto& e : bytes_of) if (e.first == t.type) { e.second += nb; found = true; break; }
      if (!found) bytes_of.emplace_back(t.type, nb);
      model_bytes += nb;
    }
    double weighted_ms = 0, covered = 0;
    printf("\n%-10s %10s %8s %10s\n", "type", "MiB", "share", "GB/s");
    for (auto& e : bytes_of) {
      double r = 0;
      for (auto& m : rate) if (m.first == e.first) { r = m.second; break; }
      printf("%-10s %10.1f %7.1f%% %10s\n", qwen::ggml_type_name(e.first),
             e.second / (1 << 20), 100.0 * e.second / model_bytes,
             r > 0 ? "" : "not sampled");
      if (r > 0) { weighted_ms += e.second / (r * 1e9) * 1e3; covered += e.second; }
    }
    if (weighted_ms > 0)
      printf("\ncomposition-weighted over the %.0f%% of model bytes whose type was "
             "sampled: %.1f GB/s\n", 100.0 * covered / model_bytes,
             covered / (weighted_ms * 1e-3) / 1e9);
  }
  printf("AWQ INT4 g128 reference: 769.8 GB/s sampled aggregate = 84.2%%\n");
  return 0;
}
