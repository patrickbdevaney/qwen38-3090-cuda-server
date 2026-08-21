// Fused GEMV over GGUF blocks: y[out] = W @ x, reading the quantised blocks
// directly with no dequantised round trip.
//
// Why fused matters. Decode is bandwidth bound: the whole point of a smaller
// quant is reading fewer bytes per token. Dequantising a tensor to bf16 into a
// workspace and then calling cuBLAS reads the blocks AND writes AND re-reads
// twice their size in bf16, which is strictly worse than the format we started
// from. We already measured that path at 12.55x a decode step.
//
// Shape: ONE WARP PER OUTPUT ROW, walking the input dimension in runs of 256
// elements. Lane l owns elements [8l, 8l+8) of every run, which for every
// format in these files lands inside a single sub-block, so each lane needs one
// scale and a handful of contiguous bytes. The 32 lanes of a warp together
// cover one super-block's worth of contiguous memory.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>
#include "gguf.h"

namespace qwen {

// One quantised weight matrix living on the device in its GGUF block format.
// Row-major [out_f, in_f]; GGUF stores ne = {in_f, out_f} which is the same
// memory.
struct GgufWeight {
  const void* data = nullptr;   // device
  GgmlType type = GgmlType::F32;
  int out_f = 0, in_f = 0;
  size_t bytes = 0;
};

// ALIGNMENT CONTRACT: w.data must be at least 16-byte aligned. Every allocation
// from cudaMalloc satisfies this, and it is what lets the Q4_K and Q5_K paths
// read their eight quantised bytes as one 64-bit load instead of eight 1-byte
// loads -- those two block formats are 144 and 176 bytes, both multiples of 16,
// so if the tensor base is aligned then every block in it is. Formats whose
// block size is not a multiple of 8 (Q2_K 84, Q3_K 110, Q6_K 210) cannot do
// this without a repack and still use byte loads. gguf_gemv() checks the
// pointer and aborts rather than issuing a misaligned load.
//
// y[out_f] = W @ x[in_f]. x and y are bf16.
void gguf_gemv(__nv_bfloat16* y, const GgufWeight& w, const __nv_bfloat16* x,
               cudaStream_t stream = 0);

// Y[M, out_f] = X[M, in_f] @ W^T for small M, same kernel with M accumulators
// so the weight stream is read once rather than M times.
void gguf_gemm_small(__nv_bfloat16* y, const GgufWeight& w, const __nv_bfloat16* x,
                     int M, cudaStream_t stream = 0);

// True if the fused path implements this type.
bool gguf_gemv_supported(GgmlType t);

// Debug entry point for gate_gguf_gemv: fill `n` values using the SAME per-lane
// deq8 path the GEMV uses, so it can be compared bit-for-bit against the
// full-block dequantiser that is already gated against ggml. Two-level
// verification: deq8 == our dequant == ggml.
void gguf_deq8_dump(float* dst, const void* src, GgmlType t, int64_t n,
                    cudaStream_t stream = 0);

// Bytes one row of `in_f` elements occupies in this format.
size_t gguf_row_bytes(GgmlType t, int in_f);

}  // namespace qwen
