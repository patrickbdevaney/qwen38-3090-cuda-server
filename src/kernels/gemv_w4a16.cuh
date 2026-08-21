// Batch-1 W4A16 GEMV: the single hottest kernel in the server.
//
// Every decoded token reads ~12.4 GiB of weights, so this kernel IS the decode
// roofline. Measured DRAM streaming read on this card is 914.2 GB/s (68.5 tok/s
// ceiling); the gate is >= 80% of that.
//
// DESIGN (v2). The obvious mapping -- one warp per output row, lanes striding
// along the input dimension -- measured 41-47% of peak. Two reasons, both fixed
// here:
//
//   1. Every lane needed DIFFERENT activations, so the x reads out of shared
//      memory were 8- to 16-way bank conflicted.
//   2. ~5 ALU ops per weight (shift, and, int->float, bf16->float, fma), and the
//      converts alone cost more than the memory traffic they served.
//
// v2 transposes the assignment: LANE l OWNS OUTPUT ROW o0+l, and the whole warp
// walks the input dimension together. Now
//   * every lane reads the SAME activation -> a broadcast, never a conflict, and
//     x can stay in L2 instead of shared memory, freeing occupancy entirely;
//   * each lane accumulates its own row, so there is no final warp reduction;
//   * weights must be interleaved so the 32 rows a warp owns are adjacent in
//     memory. That is what awq_repack does, and it is why decode needs a repack
//     at all.
//
// Dequant uses the bf16 magic-number trick instead of int->float: bf16 0x4300 is
// exactly 128.0, and its mantissa ulp at that exponent is 1, so
//     bits = 0x4300 | n   ==   the exact value 128 + n   for n in [0, 15]
// Four LOP3s turn one u32 into four bf16x2 pairs, i.e. 8 weights, and
// __bfloat1622float2 widens each pair in one instruction. That is ~2.5 ops per
// weight against ~5 before.
//
// The +128 folds into the zero point for free. The stored nibbles are
// q_stored = q + 8 and zp_stored = zp + 8, so the offsets cancel:
//     w = (q_stored - zp_stored) * s
// and with f = 128 + q_stored,
//     w = (f - (128 + zp_stored)) * s
// so per group we need only sum(f*x) and sum(x), the latter precomputed once.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace qwen {

// Weights interleaved for the lane-owns-a-row mapping.
//   qweight [out/32][in/8][32] u32   - lane l of warp b reads word t at
//                                      [b][t][l], so a warp instruction covers
//                                      128 contiguous bytes
//   scale   [out/32][G][32]    bf16
//   zp8     [out/32][G][32]    u8    - the raw stored nibble, still offset by 8
struct W4A16Weights {
  uint32_t*      qweight = nullptr;
  __nv_bfloat16* scale   = nullptr;
  uint8_t*       zp      = nullptr;
  int out_f = 0, in_f = 0, group_size = 0, num_groups = 0;

  size_t qweight_bytes() const { return size_t(out_f) * in_f / 8 * sizeof(uint32_t); }
  size_t scale_bytes()   const { return size_t(out_f) * num_groups * sizeof(__nv_bfloat16); }
  size_t zp_bytes()      const { return size_t(out_f) * num_groups; }
  size_t total_bytes()   const { return qweight_bytes() + scale_bytes() + zp_bytes(); }
};

// Build the interleaved layout from the on-disk compressed-tensors tensors.
// Runs once at load. Device pointers in, device pointers out.
void awq_repack(W4A16Weights& dst,
                const uint32_t* src_packed,   // [out][in/8]   u32
                const __nv_bfloat16* src_scale,  // [out][G]   bf16
                const uint32_t* src_zp,       // [out/8][G]    u32
                int out_f, int in_f, int group_size,
                cudaStream_t stream = 0);

// Allocate a fused destination that several source tensors sharing the same
// input will be concatenated into along the output dimension.
//
// The model has three such groups, and fusing them is a real win rather than a
// benchmark trick: q/k/v all read the same normed hidden state, gate/up both
// read the same post-attention norm, and the GDN in_proj_qkv/in_proj_z pair
// shares an input too. Fusing turns six launches into three and, more
// importantly, folds k_proj and v_proj -- 1024x5120 tensors that run at ~32% of
// peak because they are too small to fill the GPU -- into a 14336x5120 tensor
// that runs at ~80%.
//
// Every constituent out_f in this model is a multiple of 32, so the 32-row
// interleave never straddles a block boundary and concatenation is just a row
// offset.
void awq_alloc_fused(W4A16Weights& dst, int total_out_f, int in_f, int group_size,
                     cudaStream_t stream = 0);
void awq_repack_into(W4A16Weights& dst, int dst_row_offset,
                     const uint32_t* src_packed, const __nv_bfloat16* src_scale,
                     const uint32_t* src_zp, int out_f, cudaStream_t stream = 0);
void awq_free(W4A16Weights& w);

// Scratch the GEMV needs: x widened to fp32 and per-group sums of x.
// Sized for the widest in_f the model uses.
struct GemvScratch {
  float* xf     = nullptr;   // [in_f]
  float* xgsum  = nullptr;   // [num_groups]
  float* partial= nullptr;   // [max_splits][out_f]
  int    max_in = 0, max_groups = 0, max_out = 0, max_splits = 0;
  size_t bytes() const {
    return size_t(max_in) * 4 + size_t(max_groups) * 4 +
           size_t(max_splits) * max_out * 4;
  }
};

void gemv_scratch_alloc(GemvScratch& s, int max_in, int max_out, int min_group);
void gemv_scratch_free(GemvScratch& s);

// y[out] = W @ x. y may be bf16 or fp32.
void gemv_w4a16(__nv_bfloat16* y, const W4A16Weights& w, const __nv_bfloat16* x,
                GemvScratch& s, cudaStream_t stream = 0);
void gemv_w4a16_f32(float* y, const W4A16Weights& w, const __nv_bfloat16* x,
                    GemvScratch& s, cudaStream_t stream = 0);

// Exposed for the bench: how many input-splits the heuristic picks.
int gemv_choose_splits(int out_f, int num_groups);

}  // namespace qwen

namespace qwen {

// INT8 group-quantized weights, for lm_head.
//
// lm_head is the one tensor where 4 bits measurably costs output quality: it
// maps a 5120-dim state onto 248,320 logits, and a rare token's logit is carried
// by a handful of large weights that 16 levels cannot represent. INT4 measured a
// KL of 1.1e-2 (g32) to 1.8e-2 (g128) against 1.5e-3 for everything else.
// INT8 costs 1.20 GiB against BF16's 2.37 and INT4 g128's 0.62.
struct W8A16Weights {
  int8_t*        qweight = nullptr;   // [out/32][in][32] interleaved, same idea as W4A16
  __nv_bfloat16* scale   = nullptr;   // [out/32][G][32]
  int out_f = 0, in_f = 0, group_size = 0, num_groups = 0;
  size_t total_bytes() const {
    return size_t(out_f) * in_f + size_t(out_f) * num_groups * 2;
  }
};

void quantize_w8a16(W8A16Weights& dst, const __nv_bfloat16* src, int out_f, int in_f,
                    int group_size, cudaStream_t stream = 0);
void gemv_w8a16(__nv_bfloat16* y, const W8A16Weights& w, const __nv_bfloat16* x,
                GemvScratch& s, cudaStream_t stream = 0);
void w8a16_free(W8A16Weights& w);

}  // namespace qwen
