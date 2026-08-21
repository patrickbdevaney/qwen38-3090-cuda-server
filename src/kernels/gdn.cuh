// GatedDeltaNet: 48 of the model's 64 layers, so most of the model.
//
// Per token, per value head h (48 of them, each with a [128 k][128 v] state S):
//     g_t   = exp( -exp(A_log[h]) * softplus(a[h] + dt_bias[h]) )     (fp32)
//     beta  = sigmoid(b[h])
//     q, k  = l2norm(.), then q *= 1/sqrt(128)
//     S    *= g_t
//     delta = (v_t - S^T k) * beta
//     S    += k (outer) delta
//     out   = S^T q
//
// The naive form needs two passes over S: delta depends on a full reduction over
// k, and out depends on the UPDATED S. Expanding the last line,
//     out = (g*S + k (outer) delta)^T q  =  g*(S^T q) + delta * (k . q)
// so out never needs the updated S. One pass computes A = S^T k, B = S^T q and
// the scalar k.q; the writeback then produces the new S. Holding the tile in
// registers (decode) or shared memory (prefill) across the two makes it ONE read
// and ONE write of S, which is what the traffic budget assumes.
//
// State traffic is 48 layers x 48 heads x 128 x 128 x 4 B x 2 = 302 MB per
// decoded token, ~2.4% on top of the 12.4 GiB weight stream.
//
// Prefill uses the same recurrence rather than the reference's chunked matmul
// form: a block holds one head's state in shared memory (64 KB of the 99 KB
// budget) and walks the chunk sequentially. At 4096 tokens that is ~46 ms
// against ~2.9 s of prefill GEMM, ~1.6%, so the chunked form buys nothing here
// and is not worth its complexity or its extra failure modes.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace qwen {

struct GdnDims {
  int hidden      = 5120;
  int num_v_heads = 48;
  int num_k_heads = 16;
  int head_k      = 128;
  int head_v      = 128;
  int conv_k      = 4;
  float rms_eps   = 1e-6f;
  int key_dim()  const { return num_k_heads * head_k; }      // 2048
  int val_dim()  const { return num_v_heads * head_v; }      // 6144
  int conv_dim() const { return key_dim() * 2 + val_dim(); } // 10240
  int hv_ratio() const { return num_v_heads / num_k_heads; } // 3
};

// Causal depthwise conv over the concatenated q|k|v, then SiLU.
// Decode consumes and updates a [conv_dim, conv_k-1] rolling state; prefill
// writes the trailing conv_k-1 columns as the new state.
// `in_stride` is the row stride of `in`, which is the FUSED qkv|z projection
// width, not conv_dim. Passing conv_dim here reads z as if it were qkv.
void gdn_conv(__nv_bfloat16* out, float* conv_state, const __nv_bfloat16* in,
              const __nv_bfloat16* weight, int conv_dim, int conv_k, int T,
              bool use_state, int in_stride, cudaStream_t stream = 0);

// beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias). Both fp32, because
// the reference is explicit that A can overflow to -inf in fp16 otherwise.
void gdn_gates(float* g, float* beta, const __nv_bfloat16* a, const __nv_bfloat16* b,
               const __nv_bfloat16* A_log, const __nv_bfloat16* dt_bias,
               int T, int num_v_heads, cudaStream_t stream = 0);

// The recurrence. `qkv` is the post-conv [T, conv_dim] tensor; q and k are read
// with 16 heads and broadcast to 48 by index (k head = v head / 3), which avoids
// materialising the reference's repeat_interleave.
void gdn_scan(__nv_bfloat16* out,        // [T, val_dim]
              float* state,              // [num_v_heads, head_k, head_v] fp32, in/out
              const __nv_bfloat16* qkv,  // [T, conv_dim]
              const float* g, const float* beta,   // [T, num_v_heads]
              const GdnDims& d, int T, cudaStream_t stream = 0);

// Gated RMSNorm over head_v then SiLU gate: out = w * rms(x) * silu(z).
// `z_stride` likewise: z lives at column conv_dim of the fused projection.
void gdn_norm_gate(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* z,
                   const __nv_bfloat16* w, int T, int num_v_heads, int head_v,
                   float eps, int z_stride, cudaStream_t stream = 0);

}  // namespace qwen
