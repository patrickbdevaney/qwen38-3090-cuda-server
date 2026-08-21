// The model: weights, KV/GDN state, and the forward pass.
//
// Everything is allocated up front from measured sizes and the budget is
// enforced at startup. The directive is explicit about why: no lazy allocation,
// and no OOM at token 40,000.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include "../config/model_shape.h"
#include "../kernels/gemv_w4a16.cuh"
#include "../kernels/gemm_w4a16.cuh"
#include "../kernels/gdn.cuh"
#include "../kernels/attn.cuh"

namespace qwen {

struct LayerWeights {
  bool is_attn = false;
  // shared
  const __nv_bfloat16* input_ln = nullptr;
  const __nv_bfloat16* post_ln  = nullptr;
  W4A16Weights mlp_gate_up;      // [2*inter, hidden], gate|up concatenated
  W4A16Weights mlp_down;         // [hidden, inter]
  // gated delta net
  W4A16Weights gdn_in_qkvz;      // [conv_dim + val_dim, hidden], qkv|z concatenated
  W4A16Weights gdn_out;          // [hidden, val_dim]
  const __nv_bfloat16* gdn_a = nullptr;      // [num_v_heads, hidden] bf16
  const __nv_bfloat16* gdn_b = nullptr;
  const __nv_bfloat16* gdn_conv_w = nullptr; // [conv_dim, conv_k]
  const __nv_bfloat16* gdn_A_log = nullptr;
  const __nv_bfloat16* gdn_dt_bias = nullptr;
  const __nv_bfloat16* gdn_norm = nullptr;   // [head_v]
  // attention
  W4A16Weights attn_qkv;         // [q_proj_out + 2*kv_proj_out, hidden]
  W4A16Weights attn_o;           // [hidden, num_q_heads*head_dim]
  const __nv_bfloat16* q_norm = nullptr;
  const __nv_bfloat16* k_norm = nullptr;
  int attn_slot = -1;            // index into the KV cache, -1 for GDN layers
};

struct Model {
  ModelShape shape;
  GdnDims  gdn;
  AttnDims attn;
  std::vector<LayerWeights> layers;

  // vocabulary
  int8_t*  embed_q = nullptr;        // [vocab, hidden] int8
  __nv_bfloat16* embed_bf = nullptr; // used when --embed bf16
  bool embed_quantized = true;
  float*   embed_scale = nullptr;    // [vocab] fp32
  const __nv_bfloat16* final_norm = nullptr;
  __nv_bfloat16* lm_head_bf16 = nullptr;   // used when --lm-head bf16
  W4A16Weights   lm_head_q;                // --lm-head int4
  W8A16Weights   lm_head_q8;               // --lm-head int8

  // state
  uint8_t* k_cache = nullptr;        // [attn_layers][max_ctx][kv_heads][head_dim] e4m3
  uint8_t* v_cache = nullptr;
  float*   gdn_state = nullptr;      // [gdn_layers][v_heads][dk][dv] fp32
  float*   gdn_conv  = nullptr;      // [gdn_layers][conv_dim][conv_k-1] fp32
  int      max_ctx = 0;
  int      ctx_len = 0;

  // scratch
  GemvScratch  gemv;
  GemmWorkspace gemm;
  cublasHandle_t cublas = nullptr;
  cudaStream_t stream = nullptr;   // all kernels go here so capture is a no-op change
  __nv_bfloat16 *h = nullptr, *h2 = nullptr, *proj = nullptr, *mlp_tmp = nullptr;
  __nv_bfloat16 *q_buf = nullptr, *gate_buf = nullptr, *attn_out = nullptr;
  __nv_bfloat16 *gdn_qkv = nullptr, *gdn_core = nullptr, *gdn_ab = nullptr;
  __nv_bfloat16 *logits = nullptr;
  float *gdn_g = nullptr, *gdn_beta = nullptr;
  float *cos_tab = nullptr, *sin_tab = nullptr, *attn_ws = nullptr;
  float *prefill_scores = nullptr;
  __nv_bfloat16 *kv_deq = nullptr;
  int32_t *pos_buf = nullptr, *id_buf = nullptr, *argmax_scratch = nullptr;

  // DFlash2 taps: [max_batch][n_taps * hidden], already concatenated the way
  // the drafter's fc consumes it. tap_of[li] is the slot for layer li, or -1.
  __nv_bfloat16* taps = nullptr;
  std::vector<int> tap_of;
  int n_taps = 0;
  bool tap_enable = false;
  int max_batch = 0;
  int lm_head_bits = 16;   // 16 = bf16, 8 = INT8 g128, 4 = INT4
  // Debug: when set, the residual stream after every layer (plus the embedding
  // output at index 0) is captured here, matching the reference dump's
  // hidden_states list so a divergence localises to one layer.
  __nv_bfloat16* dbg_hidden = nullptr;   // [num_layers+1][max_batch][hidden]

  // Speculative-decoding capture hook. When set and armed, each GDN layer
  // records the recurrence inputs for the block so a partial acceptance can
  // replay them instead of redoing the forward.
  struct SpecState* spec = nullptr;

  // ---- CUDA graph for the decode step ----------------------------------
  // A decode step issues ~1400 kernels; at 2-3 us of launch latency each that
  // is ~4 ms against a ~18 ms budget, which measured as exactly the gap between
  // the kernel-time sum and the end-to-end number. The graph removes it.
  //
  // The captured graph must be position-INDEPENDENT, so the token id, the
  // position and the context length all live in device memory (d_step), and the
  // attention split count is fixed at capture so the grid shape never changes.
  // Graphs are BUCKETED by context. The attention split count has to be fixed
  // inside a graph, and sizing it for 128K wastes partial-buffer traffic at
  // short context: 164 splits x 24 heads x 256 dims x 16 layers is 64 MB per
  // token, which measured as 47.3 -> 44.8 tok/s at 4K. One graph per bucket
  // gets both ends.
  static constexpr int kMaxGraphs = 4;
  cudaGraph_t     graph[kMaxGraphs] = {};
  cudaGraphExec_t graph_exec[kMaxGraphs] = {};
  int             graph_ctx[kMaxGraphs] = {};     // upper bound of each bucket
  int             graph_splits_of[kMaxGraphs] = {};
  int             n_graphs = 0;
  cudaStream_t    capture_stream = nullptr;
  int32_t*        d_step = nullptr;      // [0]=token id, [1]=position, [2]=ctx_len
  int32_t*        h_step = nullptr;      // pinned staging for d_step
  int             graph_splits = 0;      // the count the CURRENT step uses
  bool            use_graph = true;

  // owned device allocations, freed on destruction
  std::vector<void*> owned;

  ~Model();
};

struct LoadOptions {
  int  max_ctx = 131072;
  int  max_batch = 4096;         // chunked-prefill chunk
  // lm_head precision. INT4 measured a KL of 1.1e-2 (g32) to 1.8e-2 (g128)
  // against 1.5e-3 for the rest of the pipeline, so it is not the default.
  int  lm_head_bits = 8;         // 16 | 8 | 4
  int  lm_head_group = 128;
  bool quantize_embed = true;    // INT8 rowwise; reclaims 1.183 GiB
  bool verbose = true;
};

// Loads the checkpoint, repacks, allocates state, and REFUSES TO START if the
// requested context does not fit with a 512 MB margin.
void model_load(Model& m, const std::string& model_dir, const LoadOptions& opt);

// One decode step: token id in, logits out. Advances ctx_len by 1.
void model_decode(Model& m, int32_t token_id, int position);
// Prefill T tokens starting at `position`; logits are produced for the last one.
void model_prefill(Model& m, const int32_t* ids, int T, int position);
// Same, but writes logits for EVERY position, which block verification needs.
void model_forward_all_logits(Model& m, const int32_t* ids, int T, int position,
                              __nv_bfloat16* logits_out);

// Capture the decode step as a CUDA graph. Call once after load.
void model_graph_capture(Model& m);
int  model_graph_bucket(const Model& m, int ctx);

// Greedy generation; returns the generated ids.
std::vector<int32_t> model_generate_greedy(Model& m, const std::vector<int32_t>& prompt,
                                           int max_new, int eos_id);

// Residual-stream taps for the DFlash2 drafter. The drafter's keys and values
// come from the TARGET's hidden states at a handful of layers, so those layers
// have to publish h as they go. Layout is [row][tap][hidden], i.e. already
// concatenated the way fc consumes it.
void model_enable_taps(Model& m, const std::vector<int>& layer_ids);
// The target's lm_head applied to arbitrary rows, WITHOUT the target's final
// norm. The drafter's output is already normalised by its own `norm`, and the
// reference calls _output_head(target) on it directly.
void model_apply_head(Model& m, __nv_bfloat16* out, const __nv_bfloat16* x, int T);

// INT4 group quantisation of an arbitrary [rows, cols] bf16 tensor into the
// 32-row interleaved layout the GEMV and MMA kernels read. The caller owns
// dst.{qweight,scale,zp}.
void quantize_w4a16(W4A16Weights& dst, const __nv_bfloat16* src, int rows, int cols,
                    int group);
void model_disable_taps(Model& m);

// QWEN_DEBUG_SYNC=2: print and reset the per-stage wall-clock profile.
void dbg_profile_report(const char* tag);

}  // namespace qwen
