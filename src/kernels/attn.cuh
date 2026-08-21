// Gated attention: 16 of 64 layers, head_dim 256, 24 Q heads / 4 KV heads.
//
// Shapes and quirks, all read out of modeling_qwen3_5.py rather than assumed:
//
//   * q_proj emits num_heads * head_dim * 2 = 12288, which is NOT 24 query
//     heads. It is 24 x [query(256) | gate(256)] INTERLEAVED per head. The gate
//     is applied at the very end as attn_out * sigmoid(gate), before o_proj.
//   * q_norm and k_norm are plain RMSNorm over the 256-wide head dim, applied
//     BEFORE RoPE.
//   * scaling is head_dim^-0.5 = 1/16.
//   * RoPE covers the FIRST 64 dims (rotary_dim = cos.shape[-1]); the other 192
//     pass through untouched. Confirmed from apply_rotary_pos_emb's
//     q[..., :rotary_dim] / q[..., rotary_dim:] split, not assumed.
//   * The rotation is the half-split form: with 32 frequencies,
//         q'[j]    = q[j]*cos_j - q[j+32]*sin_j
//         q'[j+32] = q[j+32]*cos_j + q[j]*sin_j
//   * mrope: freqs index i takes its position from axis T if i%3==0, H if
//     i%3==1, W if i%3==2, which is exactly sections [11,11,10] over 32
//     frequencies. FOR TEXT-ONLY INPUT ALL THREE AXES CARRY THE SAME POSITION,
//     so the interleave is a no-op -- but it is implemented generally and
//     asserted, because silently collapsing it is the directive's failure mode
//     #1: fluent output that decays at long context.
//
// KV cache is FP8 e4m3, 32 KiB per token across all 16 layers, which is what
// makes 128K context fit. sm_86 has no FP8 hardware, so conversions are
// software; that ALU cost is budgeted, not wished away.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>
#include <cublas_v2.h>

namespace qwen {

// KV cache element format, per side.
//
// FP8 e4m3 is 8 bits and costs 32 KiB/token across the 16 attention layers.
// INT4 with a per-32-group symmetric fp16 scale is 4.5 bits and costs 18
// KiB/token, which at 262144 context is 4.5 GiB instead of 8.0.
//
// The point of making K and V independent is that they are NOT equally
// sensitive: keys decide WHERE attention goes, values only what it carries, so
// K FP8 / V INT4 is the conservative step and INT4/INT4 is the aggressive one.
enum class KvFmt { FP8 = 0, INT4 = 1 };

struct AttnDims {
  KvFmt k_fmt = KvFmt::FP8;
  KvFmt v_fmt = KvFmt::FP8;
  int hidden      = 5120;
  int num_q_heads = 24;
  int num_kv_heads = 4;
  int head_dim    = 256;
  int rotary_dim  = 64;
  float rope_theta = 1e7f;
  float rms_eps   = 1e-6f;
  int q_per_kv() const { return num_q_heads / num_kv_heads; }   // 6
  int q_proj_out() const { return num_q_heads * head_dim * 2; } // 12288 (q|gate)
  int kv_proj_out() const { return num_kv_heads * head_dim; }   // 1024
  int o_proj_in() const { return num_q_heads * head_dim; }      // 6144
  float scaling() const { return 1.0f / 16.0f; }                // head_dim^-0.5

  // Bytes of quantised data per (position, kv head) for one side, and the
  // matching group-scale count. INT4 packs two dims per byte and carries one
  // fp16 scale per 32 dims.
  static int side_bytes(KvFmt f, int head_dim) {
    return f == KvFmt::FP8 ? head_dim : head_dim / 2;
  }
  static int side_scales(KvFmt f, int head_dim) {
    return f == KvFmt::FP8 ? 0 : head_dim / 32;
  }
  int k_bytes_per_pos() const { return num_kv_heads * side_bytes(k_fmt, head_dim); }
  int v_bytes_per_pos() const { return num_kv_heads * side_bytes(v_fmt, head_dim); }
  int k_scales_per_pos() const { return num_kv_heads * side_scales(k_fmt, head_dim); }
  int v_scales_per_pos() const { return num_kv_heads * side_scales(v_fmt, head_dim); }
  // Total KV bytes for one token across ONE layer, quants plus scales.
  int kv_bytes_per_pos() const {
    return k_bytes_per_pos() + v_bytes_per_pos() +
           2 * (k_scales_per_pos() + v_scales_per_pos());
  }
};

// "INT4" here is 4-bit two's complement in [-7, 7] with a per-32 symmetric
// scale, so the decode is  value = scale * sign_extend_4(nibble).
__host__ __device__ __forceinline__ int kv_i4_decode(uint32_t nibble) {
  return int((nibble ^ 8u)) - 8;
}

// Precomputed cos/sin for one position range. [T][rotary_dim/2] each.
void rope_tables(float* cos_out, float* sin_out, const int32_t* pos_t,
                 const int32_t* pos_h, const int32_t* pos_w,
                 int T, int rotary_dim, float theta, cudaStream_t stream = 0);

// Split q_proj output into query and gate, RMSNorm the query and the key, apply
// RoPE to both, and write K/V into the FP8 cache at `cache_pos`.
//  qkv_in : [T, q_proj_out + kv_proj_out + kv_proj_out] bf16 (the fused GEMV out)
//  q_out  : [T, num_q_heads, head_dim] bf16, post-norm post-rope
//  gate   : [T, num_q_heads * head_dim] bf16
// `k_scale`/`v_scale` are the per-32-group fp16 scales, used only when the
// matching side is INT4; pass null for FP8.
void attn_prepare(__nv_bfloat16* q_out, __nv_bfloat16* gate,
                  uint8_t* k_cache, uint8_t* v_cache,
                  uint16_t* k_scale, uint16_t* v_scale,
                  const __nv_bfloat16* qkv_in,
                  const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
                  const float* cos_tab, const float* sin_tab,
                  int T, int cache_pos, int max_ctx, const AttnDims& d,
                  cudaStream_t stream = 0);

// Graph-capturable variants: the position and context length are read from
// DEVICE memory instead of being baked in as kernel arguments, so one captured
// graph serves every decode step. Without this a graph would be pinned to the
// position it was captured at.
void attn_prepare_dev(__nv_bfloat16* q_out, __nv_bfloat16* gate,
                      uint8_t* k_cache, uint8_t* v_cache,
                      uint16_t* k_scale, uint16_t* v_scale,
                      const __nv_bfloat16* qkv_in,
                      const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
                      const float* cos_tab, const float* sin_tab,
                      const int32_t* d_pos, int max_ctx, const AttnDims& d,
                      cudaStream_t stream = 0);
void attn_decode_dev(__nv_bfloat16* out, const __nv_bfloat16* q,
                     const uint8_t* k_cache, const uint8_t* v_cache,
                     const uint16_t* k_scale, const uint16_t* v_scale,
                     const int32_t* d_ctx_len, int max_ctx, const AttnDims& d,
                     float* workspace, int splits, cudaStream_t stream = 0);
// Fills a [1] position buffer and the rope table for a single decode step.
void rope_tables_dev(float* cos_out, float* sin_out, const int32_t* d_pos,
                     int rotary_dim, float theta, cudaStream_t stream = 0);

// Flash-decoding: one query position, KV split across blocks.
// out: [num_q_heads, head_dim] bf16
void attn_decode(__nv_bfloat16* out, const __nv_bfloat16* q,
                 const uint8_t* k_cache, const uint8_t* v_cache,
                 const uint16_t* k_scale, const uint16_t* v_scale,
                 int ctx_len, int max_ctx, const AttnDims& d,
                 float* workspace, int splits, cudaStream_t stream = 0);
int attn_decode_splits(int ctx_len);
size_t attn_decode_workspace_bytes(const AttnDims& d, int max_splits);

// Prefill: T new queries against ctx_len keys, causal, tiled over KV with an
// online softmax so the score matrix is never materialised at full context.
void attn_prefill(__nv_bfloat16* out, const __nv_bfloat16* q,
                  const uint8_t* k_cache, const uint8_t* v_cache,
                  const uint16_t* k_scale, const uint16_t* v_scale,
                  int T, int ctx_len, int q_offset, int max_ctx,
                  const AttnDims& d, __nv_bfloat16* kv_scratch, float* score_scratch,
                  cublasHandle_t cublas, cudaStream_t stream = 0);
size_t attn_prefill_scratch_bytes(const AttnDims& d, int kv_tile, int q_tile);

// out = attn_out * sigmoid(gate), in place on attn_out.
void attn_output_gate(__nv_bfloat16* attn_out, const __nv_bfloat16* gate,
                      int n, cudaStream_t stream = 0);

}  // namespace qwen
