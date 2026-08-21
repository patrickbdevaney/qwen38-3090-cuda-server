// DFlash2 block-diffusion drafter.
//
// The drafter is a 5-layer Qwen3 model that does NOT read tokens the way a
// normal draft model does. Its keys and values come from the TARGET's residual
// stream, tapped at layers [5,19,33,47,61] and fused by fc: 5*5120 -> 5120. Its
// queries come from the "noise" block -- the last committed token followed by
// block_size-1 copies of the mask token, embedded with the TARGET's embedding
// table. One forward emits all 7 draft positions at once; there is no
// autoregressive inner loop.
//
// It also has no lm_head and no embedding of its own. Logits come from the
// TARGET's head, and the candidate selector then refines the top-16 of those
// logits into a path using two rank-256 codebooks and a first-order Markov
// chain. That is the whole point of the selector: it costs 16*256 MACs per
// position instead of another 248320-wide projection.
//
// Everything here mirrors z-lab/dflash's DFlash2DraftModel. Where the reference
// makes a choice that looks arbitrary (which RMSNorm rounds where, that the
// conv's leading base_kernel axis is prepare/finish and not a tap), this file
// follows it and says so, because gate_dflash compares against a dump driven
// through the reference itself.
#pragma once
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <string>
#include <vector>
#include "../kernels/gemv_w4a16.cuh"

namespace qwen {

struct DFlashShape {
  int hidden = 0, n_layers = 0, n_taps = 0, block_size = 0;
  int mask_token_id = 0, vocab = 0;
  int n_q_heads = 0, n_kv_heads = 0, head_dim = 0, sliding_window = 0;
  int top_k = 0, rank = 0, conv_k = 0, conv_group = 0;
  bool is_causal = false;
  float rms_eps = 1e-6f, rope_theta = 1e7f, embed_scale = 1.0f;
  std::vector<int> target_layer_ids;

  int q_dim()  const { return n_q_heads * head_dim; }
  int kv_dim() const { return n_kv_heads * head_dim; }
  int qkv_dim() const { return q_dim() + 2 * kv_dim(); }
  int conv_groups() const { return hidden / conv_group; }
  // kernel_projection emits 2 (prepare/finish) * conv_k taps * groups
  int kproj_out() const { return 2 * conv_k * conv_groups(); }
};

// One drafter layer. Weights are held either as bf16 (exact, 3.85 GB total) or
// W4A16 (about 1.2 GB); the choice only changes draft QUALITY, never output
// correctness, because every drafted token is verified by the target.
struct DraftLayer {
  __nv_bfloat16* qkv = nullptr;        // [qkv_dim, hidden] fused q|k|v
  __nv_bfloat16* o = nullptr;          // [hidden, q_dim]
  __nv_bfloat16* gate_up = nullptr;    // [2*inter, hidden] fused
  __nv_bfloat16* down = nullptr;       // [hidden, inter]
  __nv_bfloat16* attn_kproj = nullptr; // [kproj_out, hidden]
  __nv_bfloat16* mlp_kproj = nullptr;  // [kproj_out, hidden]
  W4A16Weights qkv_q, o_q, gate_up_q, down_q, attn_kproj_q, mlp_kproj_q;

  __nv_bfloat16* input_ln = nullptr;
  __nv_bfloat16* post_ln = nullptr;
  __nv_bfloat16* q_norm = nullptr;     // [head_dim]
  __nv_bfloat16* k_norm = nullptr;
  __nv_bfloat16* attn_base = nullptr;  // [2][conv_k][hidden]
  __nv_bfloat16* mlp_base = nullptr;
};

struct DraftModel {
  DFlashShape sh;
  int inter = 0;
  bool quantized = false;
  std::vector<DraftLayer> layers;

  __nv_bfloat16* fc = nullptr;          // [hidden, n_taps*hidden]
  W4A16Weights fc_q;
  __nv_bfloat16* hidden_norm = nullptr; // [hidden]
  __nv_bfloat16* final_norm = nullptr;  // [hidden]

  // Selector. Kept in bf16 regardless: the codebooks are gathers, not matmuls,
  // and hidden_projection is 2.6 MB.
  __nv_bfloat16* pred_cb = nullptr;     // [vocab, rank]
  __nv_bfloat16* succ_cb = nullptr;     // [vocab, rank]
  __nv_bfloat16* hproj = nullptr;       // [rank, hidden]

  // Context KV cache, derived from the target's residual stream. Capacity is
  // 2*sliding_window so the compaction that keeps the window resident happens
  // once every ~sliding_window tokens instead of every step.
  __nv_bfloat16* k_cache = nullptr;     // [layers][cap][kv_dim]
  __nv_bfloat16* v_cache = nullptr;
  int cache_cap = 0;                    // slots
  int cache_len = 0;                    // slots in use
  int cache_pos0 = 0;                   // absolute position of slot 0

  // Scratch, sized for ctx_chunk context rows plus one block.
  int max_rows = 0;
  __nv_bfloat16 *ctx_h = nullptr, *noise = nullptr, *h = nullptr, *h2 = nullptr;
  __nv_bfloat16 *qbuf = nullptr, *kvbuf = nullptr, *attn_out = nullptr;
  __nv_bfloat16 *proj = nullptr, *mlp_tmp = nullptr, *dynbuf = nullptr;
  __nv_bfloat16 *cbuf = nullptr;   // conv output, [block, hidden]
  float *cos_tab = nullptr, *sin_tab = nullptr;
  int32_t* pos_buf = nullptr;
  int32_t* ids_buf = nullptr;
  __nv_bfloat16* hp = nullptr;          // [block, rank]
  float* unary = nullptr;               // [block, top_k]
  int32_t* cand = nullptr;              // [block, top_k]
  int32_t* path = nullptr;              // [block]
  GemvScratch gemv;

  // Stage taps for gate_dflash. Allocated only when `debug` is set; the gate
  // compares each against the reference's own forward hooks so a mismatch names
  // the stage instead of just the output.
  bool debug = false;
  enum { DBG_CTX_NORM, DBG_L0_LN, DBG_L0_CONV, DBG_L0_ATTN, DBG_L0_POST_LN,
         DBG_L0_OUT, DBG_N };
  __nv_bfloat16* dbg[DBG_N] = {};

  cublasHandle_t cublas = nullptr;
  cudaStream_t stream = 0;
  std::vector<void*> owned;
  size_t bytes = 0;
};

struct DraftLoadOptions {
  bool quantize = false;     // W4A16 the big matmuls
  int group_size = 128;
  int ctx_chunk = 512;       // most context rows accepted in one push
  bool debug = false;        // allocate the stage taps gate_dflash reads
  bool verbose = true;
};

void draft_load(DraftModel& d, const std::string& dir, const DraftLoadOptions& opt);
void draft_free(DraftModel& d);

// Reset the context cache (new sequence).
void draft_reset(DraftModel& d);

// Run one drafter forward.
//
//   target_hidden  [n_ctx, n_taps*hidden]  residual stream at the tap layers,
//                                          for the newly committed positions
//   ctx_pos0       absolute position of target_hidden row 0
//   block_pos0     absolute position of noise row 0
//
// When `use_cache` is false the context is not retained -- that is the shape the
// reference dump uses and the shape gate_dflash checks.
//
// The caller fills d.noise ([block, hidden]) with the embedded block BEFORE
// calling: the drafter has no embedding table of its own, so the noise comes
// from the target's, scaled by input_embedding_scale.
//
// Returns all `block` output rows; the caller slices rows 1.. as the mask slots.
// Positions must be contiguous: block_pos0 == ctx_pos0 + n_ctx, and with a cache
// ctx_pos0 == cache_pos0 + cache_len.
const __nv_bfloat16* draft_forward(DraftModel& d, const __nv_bfloat16* target_hidden,
                                   int n_ctx, int ctx_pos0, int block_pos0,
                                   bool use_cache);

// Candidate selection. `logits` is [block-1, vocab] from the TARGET head over
// the drafter hidden rows. Fills d.path with block-1 token ids.
void draft_select(DraftModel& d, const __nv_bfloat16* draft_hidden,
                  const __nv_bfloat16* logits, int n_pos, int32_t anchor_id);

}  // namespace qwen
