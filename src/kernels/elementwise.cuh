// Small kernels the forward pass needs between the big ones.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace qwen {

// Qwen3_5RMSNorm: norm(x.float()) * (1.0 + weight.float()), fp32 throughout.
// This is NOT the gated variant the GDN uses, which rounds to bf16 before the
// weight and has no "1.0 +". The weights ship as zeros, so the "1.0 +" is
// load-bearing. Mixing the two up puts activations ~96% out.
void rmsnorm(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* w,
             int rows, int dim, float eps, cudaStream_t stream = 0);

// out[i] += x[i]
void residual_add(__nv_bfloat16* out, const __nv_bfloat16* x, int n,
                  cudaStream_t stream = 0);

// out = silu(gate) * up, reading a fused [rows, 2*inter] buffer laid out as
// [gate | up] per row -- the order awq concatenation produces.
void swiglu(__nv_bfloat16* out, const __nv_bfloat16* fused, int rows, int inter,
            cudaStream_t stream = 0);

// INT8 row-quantized embedding lookup. Storing the table at int8 saves 1.18 GiB
// against bf16, and it is a pure row gather, so the extra dequant is off the
// critical path.
void embed_int8(__nv_bfloat16* out, const int8_t* table, const float* row_scale,
                const int32_t* ids, int n, int dim, cudaStream_t stream = 0);
void embed_bf16(__nv_bfloat16* out, const __nv_bfloat16* table, const int32_t* ids,
                int n, int dim, cudaStream_t stream = 0);
void quantize_embed_int8(int8_t* q, float* row_scale, const __nv_bfloat16* src,
                         int rows, int dim, cudaStream_t stream = 0);

// Greedy argmax over the vocabulary, on device. A single host copy of the logit
// vector would be 993 KB per token and would serialise the pipeline; only the
// chosen id crosses the bus.
void argmax(int32_t* out_id, const __nv_bfloat16* logits, int n,
            int32_t* scratch, cudaStream_t stream = 0);

}  // namespace qwen
