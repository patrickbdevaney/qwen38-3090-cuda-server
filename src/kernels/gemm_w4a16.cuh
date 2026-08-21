// Prefill W4A16 GEMM.
//
// Prefill is COMPUTE bound, not memory bound: 8K tokens through this model is
// 399 TFLOP of GEMM against 11.9 GiB of weight traffic. At the measured 81.6
// TFLOPS bf16 peak the arithmetic takes seconds while the weights take 14 ms, so
// the only thing that matters is tensor-core efficiency.
//
// That inverts the decode picture and changes the right answer. Rather than hand
// -writing a Marlin-class fused-dequant GEMM, this dequantizes each weight tile
// to bf16 into a workspace and calls cuBLAS, which the directive explicitly
// permits for the prefill path. The dequant costs one extra write plus one extra
// read of the weights: 55.7 ms against ~8 s of GEMM at 8K, i.e. 0.7%. Hand-
// writing a GEMM to reclaim 0.7% while risking correctness would be the wrong
// trade, and it is exactly the "optimize prefill before decode is correct"
// failure mode the directive warns about.
//
// Measured llama.cpp does 1318 tok/s at pp8192, which is 64.1 TFLOPS effective
// -- 79% of this card's bf16/FP32-accumulate peak. That is too high for an
// FP32-accumulate kernel, so llama.cpp is almost certainly accumulating in FP16
// (142 TFLOPS peak on GA102, where it would be a realistic 45%). We accumulate
// in FP32 by default; FP16 accumulation is a Phase 9 option behind a KL gate.
#pragma once
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include "gemv_w4a16.cuh"

namespace qwen {

struct GemmWorkspace {
  __nv_bfloat16* wbuf = nullptr;   // dequantized weight tile
  size_t         bytes = 0;
  int            max_rows = 0;     // output rows that fit for a given in_f
  cublasHandle_t cublas = nullptr;
};

void gemm_workspace_alloc(GemmWorkspace& ws, size_t bytes);
void gemm_workspace_free(GemmWorkspace& ws);

// C[M, out_f] = A[M, in_f] @ W^T, all row-major, A and C bf16.
// Tiles over output rows so the dequantized weight always fits the workspace.
void gemm_w4a16(__nv_bfloat16* C, const W4A16Weights& w, const __nv_bfloat16* A,
                int M, GemmWorkspace& ws, cudaStream_t stream = 0);

// Dequantize the interleaved int4 weights to a plain row-major bf16 matrix.
// Exposed so the numerics gate can check it directly against the host unpack.
void w4a16_dequant_rows(__nv_bfloat16* dst, const W4A16Weights& w,
                        int row0, int nrows, cudaStream_t stream = 0);

}  // namespace qwen
