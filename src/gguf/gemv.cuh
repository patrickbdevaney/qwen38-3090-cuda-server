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
// `ldy` is the row stride of y in elements; 0 means "tightly packed", i.e.
// w.out_f. A non-default stride lets a projection that GGUF ships as several
// separate tensors (q/k/v, or GDN's qkv and z) be written into consecutive
// column ranges of one [T, total_out] buffer, which is the layout the AWQ path
// gets for free from its single fused tensor.
// INT8 ACTIVATIONS (W4A8).
//
// The i-quants are ALU bound, not bandwidth bound: Q6_K, whose dequantiser is
// trivial, runs at 89% of measured DRAM while IQ4_XS and IQ3_S -- 65% of a
// UD-Q3_K_XL file -- sit at 67% because every value costs a codebook lookup, an
// int-to-float conversion and an FMA. Quantising the ACTIVATION vector to int8
// once per projection turns the inner product into __dp4a: four multiply-adds
// per instruction, against eight conversions plus eight FMAs.
//
// This changes the arithmetic, so it is opt-in per type (Deq<T>::DP4A) and its
// effect on KL against the BF16 reference is measured, not assumed.
//
// `qx` holds in_f int8 values per row and `xsc` one fp32 scale per 32 of them.
// A lane owns eight consecutive elements, which always lie inside one group.
void gguf_quantize_x(int8_t* qx, float* xsc, const __nv_bfloat16* x, int in_f, int M,
                     cudaStream_t stream = 0);
// Scratch needed by gguf_quantize_x for in_f elements and M rows.
inline size_t gguf_qx_bytes(int in_f, int M) { return size_t(in_f) * M + size_t(in_f / 32) * M * 4; }
// True if any type in this build uses the int8 activation path.
bool gguf_wants_qx(GgmlType t);

void gguf_gemv(__nv_bfloat16* y, const GgufWeight& w, const __nv_bfloat16* x,
               cudaStream_t stream = 0, int ldy = 0,
               const int8_t* qx = nullptr, const float* xsc = nullptr);

// Y[M, out_f] = X[M, in_f] @ W^T for small M, same kernel with M accumulators
// so the weight stream is read once rather than M times.
void gguf_gemm_small(__nv_bfloat16* y, const GgufWeight& w, const __nv_bfloat16* x,
                     int M, cudaStream_t stream = 0, int ldy = 0,
                     const int8_t* qx = nullptr, const float* xsc = nullptr);

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
