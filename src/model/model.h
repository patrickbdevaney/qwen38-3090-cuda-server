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
#include "../kernels/attn.cuh"
#include "../kernels/gemm_w4a16.cuh"
#include "../kernels/gdn.cuh"
#include "../gguf/gemv.cuh"
#include "../kernels/attn.cuh"

namespace qwen {

// One projection, in whichever weight format the checkpoint uses.
//
// AWQ ships q|k|v (and GDN's qkv|z, and gate|up) already concatenated into a
// single packed tensor, so the AWQ path gets a [T, total_out] result from one
// call. GGUF ships them as SEPARATE tensors, each free to carry its own quant
// type -- blk.3 of UD-Q3_K_XL has attn_q as IQ4_NL, attn_k as Q4_K and attn_v
// as Q5_K -- so they can neither be concatenated at load nor read by one
// kernel. This holds either shape: `awq` for the fused case, or up to three
// GGUF pieces written into consecutive column ranges of the same output buffer
// via the kernels' ldy stride.
struct Linear {
  W4A16Weights awq;              // valid when !gguf
  GgufWeight   part[3];          // valid when gguf, in output order
  int          n_part = 0;
  bool         gguf = false;
  int          in_f = 0, out_f = 0;   // total, summed over parts

  bool valid() const { return gguf ? n_part > 0 : awq.qweight != nullptr; }
  size_t total_bytes() const {
    if (!gguf) return awq.total_bytes();
    size_t b = 0;
    for (int i = 0; i < n_part; ++i) b += part[i].bytes;
    return b;
  }
};

struct LayerWeights {
  bool is_attn = false;
  // shared
  const __nv_bfloat16* input_ln = nullptr;
  const __nv_bfloat16* post_ln  = nullptr;
  Linear mlp_gate_up;      // [2*inter, hidden], gate|up concatenated
  Linear mlp_down;         // [hidden, inter]
  // gated delta net
  Linear gdn_in_qkvz;      // [conv_dim + val_dim, hidden], qkv|z concatenated
  Linear gdn_out;          // [hidden, val_dim]
  const __nv_bfloat16* gdn_a = nullptr;      // [num_v_heads, hidden] bf16
  const __nv_bfloat16* gdn_b = nullptr;
  const __nv_bfloat16* gdn_conv_w = nullptr; // [conv_dim, conv_k]
  const __nv_bfloat16* gdn_A_log = nullptr;
  const __nv_bfloat16* gdn_dt_bias = nullptr;
  const __nv_bfloat16* gdn_norm = nullptr;   // [head_v]
  // attention
  Linear attn_qkv;         // [q_proj_out + 2*kv_proj_out, hidden]
  Linear attn_o;           // [hidden, num_q_heads*head_dim]
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
  // Host-resident embedding (opt.embed_host). `embed_host_bf` is plain host
  // memory holding the whole table; `embed_stage` is a pinned staging buffer the
  // gathered rows are assembled into before one DMA.
  __nv_bfloat16* embed_host_bf = nullptr;
  __nv_bfloat16* embed_stage = nullptr;
  bool embed_on_host = false;

  int8_t*  embed_q = nullptr;        // [vocab, hidden] int8
  __nv_bfloat16* embed_bf = nullptr; // used when --embed bf16
  bool embed_quantized = true;
  float*   embed_scale = nullptr;    // [vocab] fp32
  const __nv_bfloat16* final_norm = nullptr;
  __nv_bfloat16* lm_head_bf16 = nullptr;   // used when --lm-head bf16
  W4A16Weights   lm_head_q;                // --lm-head int4
  W8A16Weights   lm_head_q8;               // --lm-head int8
  Linear         lm_head_gg;               // GGUF checkpoint: output.weight, in blocks

  // state
  uint8_t* k_cache = nullptr;        // [attn_layers][max_ctx][kv_heads][bytes/head]
  uint8_t* v_cache = nullptr;
  // Per-32-group fp16 scales, allocated only for a side that is INT4.
  uint16_t* k_scale = nullptr;       // [attn_layers][max_ctx][kv_heads][head_dim/32]
  uint16_t* v_scale = nullptr;
  float*   gdn_state = nullptr;      // [gdn_layers][v_heads][dk][dv] fp32
  float*   gdn_conv  = nullptr;      // [gdn_layers][conv_dim][conv_k-1] fp32
  int      max_ctx = 0;
  int      ctx_len = 0;

  // Prefill scratch for GGUF weights: one dequantised bf16 projection.
  // Decode reads the blocks directly, but prefill cannot -- the fused GEMV tops
  // out at 8 rows, so a 4096-token chunk would stream every weight 512 times.
  // Dequantising one tensor into this buffer and handing it to cuBLAS reads it
  // ONCE. Sized for the largest single part in the file (5120 x 17408 = 178 MiB
  // in this checkpoint) and allocated only when a GGUF is loaded.
  __nv_bfloat16* gguf_deq = nullptr;
  size_t         gguf_deq_elems = 0;

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
  // mrope: three position axes. Text-only prefill passes the same array three
  // times, which is the identity; an image span sets them independently.
  int32_t *pos_t = nullptr, *pos_h = nullptr, *pos_w = nullptr;

  // With images, the mrope position and the KV slot diverge: an image advances
  // mrope by max(h, w) while occupying h*w slots. transformers calls the gap
  // mrope_position_deltas and adds it to every later position. Zero for
  // text-only prompts, which is why the text path never had to care.
  int mrope_delta = 0;

  __nv_bfloat16* taps = nullptr;
  std::vector<int> tap_of;
  int n_taps = 0;
  bool tap_enable = false;
  int max_batch = 0;
  int lm_head_bits = 16;   // 16 = bf16, 8 = INT8 g128, 4 = INT4, 0 = GGUF blocks
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
  // Load the WEIGHTS from this GGUF file instead of the directory's
  // safetensors. The directory is still read, for config.json and
  // tokenizer.json -- a GGUF carries equivalent metadata but this server's
  // shape parser and tokenizer are gated against the HF files, and reusing them
  // keeps one source of truth for the shape.
  std::string gguf;
  int  max_ctx = 131072;
  int  max_batch = 4096;         // chunked-prefill chunk
  // lm_head precision. INT4 measured a KL of 1.1e-2 (g32) to 1.8e-2 (g128)
  // against 1.5e-3 for the rest of the pipeline, so it is not the default.
  int  lm_head_bits = 8;         // 16 | 8 | 4
  int  lm_head_group = 128;
  bool quantize_embed = true;    // INT8 rowwise; reclaims 1.183 GiB
  // Keep embed_tokens in HOST memory and DMA one row per token instead. The
  // embedding is a pure row gather -- it is never a matmul -- so the only cost
  // is 10 KB of PCIe per decoded token, against 1.185 GiB of device memory
  // reclaimed. That 1.185 GiB is 38k tokens of FP8 KV, which is the difference
  // between 128K and 262K context on a 24 GB card. Held in bf16, so unlike
  // INT8-on-device it costs no accuracy at all.
  bool embed_host = false;
  // KV cache element format, per side. FP8 e4m3 costs 32 KiB/token across the
  // 16 attention layers; INT4 with a per-32-group scale costs 18 KiB, which at
  // 262144 context is 4.5 GiB instead of 8.0. K and V are independent because
  // they are not equally sensitive: keys decide WHERE attention goes.
  KvFmt kv_k = KvFmt::FP8;
  KvFmt kv_v = KvFmt::FP8;
  bool verbose = true;
};

// Loads the checkpoint, repacks, allocates state, and REFUSES TO START if the
// requested context does not fit with a 512 MB margin.
void model_load(Model& m, const std::string& model_dir, const LoadOptions& opt);

// Fill an already-shaped Model's per-layer weights from a GGUF file, returning
// the body's device bytes. Defined in load_gguf.cu; see that file for how the
// two formats' fusing and naming differ.
class GgufFile;
struct GgufTensor;
size_t model_load_gguf_weights(Model& m, GgufFile& f, bool verbose);
GgufWeight gguf_upload_weight(Model& m, const GgufTensor& t);

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
// Gather T embedding rows from the host table into m.h. Only valid when
// m.embed_on_host.
void embed_rows_host(Model& m, const int32_t* ids, int T);
// Same gather, into an arbitrary destination. The drafter needs it: its "noise"
// input is a row of the TARGET's embedding table, and with --embed-host that
// table is not on the device at all.
void embed_rows_host_into(Model& m, const int32_t* ids, int T, __nv_bfloat16* dst,
                          cudaStream_t st);

// A run of precomputed embedding rows (image tokens) to drop into the hidden
// state after the token embedding, replacing whatever the placeholder id
// produced.
struct EmbedSplice {
  int dst_row = 0;                        // row within THIS prefill chunk
  int n_rows = 0;
  const __nv_bfloat16* src = nullptr;     // [n_rows, hidden] on device
};

// Prefill with explicit 3-axis mrope positions and optional embedding splices.
// model_prefill() is this with sequential positions on all three axes and no
// splices.
void model_prefill_mm(Model& m, const int32_t* ids, int T, int position,
                      const int32_t* pt, const int32_t* ph, const int32_t* pw,
                      const EmbedSplice* splices, int n_splices);

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
