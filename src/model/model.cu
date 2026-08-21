#include "model.h"
#include "../loader/safetensors.h"
#include "../kernels/elementwise.cuh"
#include "../spec/spec.h"

#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cuda_runtime.h>

namespace qwen {
namespace {

#define CKM(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); abort(); } } while(0)

__global__ void k_capture_pre(__nv_bfloat16* __restrict__ dst,
                              const __nv_bfloat16* __restrict__ src,
                              int T, int conv_dim, int stride) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(T) * conv_dim) return;
  dst[i] = src[(i / conv_dim) * stride + (i % conv_dim)];
}

void capture_pre_conv(__nv_bfloat16* dst, const __nv_bfloat16* src, int T, int conv_dim,
                      int stride, cudaStream_t st) {
  const size_t n = size_t(T) * conv_dim;
  k_capture_pre<<<(n + 255) / 256, 256, 0, st>>>(dst, src, T, conv_dim, stride);
}

// Debug aid: QWEN_DEBUG_SYNC=1 synchronises after every stage of run_layer and
// aborts at the first kernel that faults, instead of letting the sticky error
// surface hundreds of launches later in an unrelated call.
cudaStream_t g_dbg_stream = cudaStreamPerThread;

// QWEN_DEBUG_SYNC=1 aborts at the first faulting stage. =2 turns the same hooks
// into a per-stage wall-clock profile summed over all 64 layers, which is how
// the "T>=2 costs a flat 40 ms" cliff was traced to a single kernel.
struct DbgStage { const char* name; double ms; long n; };
DbgStage g_dbg[32];
int g_dbg_n = 0;
double g_dbg_last = 0;

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

void dsync(const char* where, int li = -1) {
  static int on = -1;
  if (on < 0) { const char* e = getenv("QWEN_DEBUG_SYNC"); on = e ? atoi(e) : 0; }
  if (!on) return;
  // cudaDeviceSynchronize is illegal mid-capture; the debug sweep has to sit out
  // the graph build.
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(g_dbg_stream, &cs) == cudaSuccess &&
      cs != cudaStreamCaptureStatusNone) { cudaGetLastError(); return; }
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    fprintf(stderr, "QWEN_DEBUG_SYNC: fault at layer %d stage %s -> %s\n", li, where,
            cudaGetErrorString(e));
    abort();
  }
  if (on < 2) return;
  const double t = now_ms();
  int slot = -1;
  for (int i = 0; i < g_dbg_n; ++i) if (g_dbg[i].name == where) { slot = i; break; }
  if (slot < 0 && g_dbg_n < 32) { slot = g_dbg_n++; g_dbg[slot] = {where, 0.0, 0}; }
  if (slot >= 0 && g_dbg_last > 0) { g_dbg[slot].ms += t - g_dbg_last; g_dbg[slot].n++; }
  g_dbg_last = t;
}

// dst[r][off .. off+H) = src[r][0 .. H)
__global__ void k_tap(__nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src,
                      int T, int H, int dst_stride, int off) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(T) * H) return;
  const int r = int(i / H), c = int(i % H);
  dst[size_t(r) * dst_stride + off + c] = src[size_t(r) * H + c];
}

void* dalloc(Model& m, size_t bytes) {
  void* p = nullptr;
  CKM(cudaMalloc(&p, bytes));
  m.owned.push_back(p);
  return p;
}

const __nv_bfloat16* upload(Model& m, const SafeTensors& st, const std::string& name) {
  const auto& t = st.get(name);
  void* d = dalloc(m, t.nbytes);
  CKM(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
  return static_cast<const __nv_bfloat16*>(d);
}

// Repack one or more on-disk quantized tensors into a single fused W4A16.
void fuse(Model& m, const SafeTensors& st, W4A16Weights& dst,
          const std::vector<std::string>& parts, int in_f, int group) {
  int total = 0;
  for (const auto& b : parts) total += int(st.get(b + ".weight_scale").shape[0]);
  awq_alloc_fused(dst, total, in_f, group);
  m.owned.push_back(dst.qweight); m.owned.push_back(dst.scale); m.owned.push_back(dst.zp);
  int off = 0;
  for (const auto& b : parts) {
    const auto& p = st.get(b + ".weight_packed");
    const auto& s = st.get(b + ".weight_scale");
    const auto& z = st.get(b + ".weight_zero_point");
    void *dp, *ds, *dz;
    CKM(cudaMalloc(&dp, p.nbytes)); CKM(cudaMemcpy(dp, p.data, p.nbytes, cudaMemcpyHostToDevice));
    CKM(cudaMalloc(&ds, s.nbytes)); CKM(cudaMemcpy(ds, s.data, s.nbytes, cudaMemcpyHostToDevice));
    CKM(cudaMalloc(&dz, z.nbytes)); CKM(cudaMemcpy(dz, z.data, z.nbytes, cudaMemcpyHostToDevice));
    awq_repack_into(dst, off, static_cast<const uint32_t*>(dp),
                    static_cast<const __nv_bfloat16*>(ds),
                    static_cast<const uint32_t*>(dz), int(s.shape[0]));
    CKM(cudaDeviceSynchronize());
    cudaFree(dp); cudaFree(ds); cudaFree(dz);
    off += int(s.shape[0]);
  }
}

// Quantize a bf16 [rows, cols] matrix to INT4 g128 asymmetric, then repack into
// the GEMV's interleaved layout. Used for lm_head, which ships bf16 and is
// 2.368 GiB -- 1.753 GiB of which this reclaims.
// Quantize a bf16 [rows, cols] matrix to INT4 group-asymmetric, then repack into
// the GEMV's interleaved layout. Used for lm_head, which ships bf16.
//
// NOTE: INT4 here measured a KL of 1.1e-2 (g32) to 1.8e-2 (g128) against the
// reference, against 1.5e-3 for the rest of the pipeline combined -- the
// quantizer dominates. lm_head does NOT ship at INT4; see PHASE_5.md. This path
// is kept for the ablation.
__global__ void k_quant_i4(uint32_t* __restrict__ packed, __nv_bfloat16* __restrict__ scale,
                           uint32_t* __restrict__ zp, const __nv_bfloat16* __restrict__ src,
                           int cols, int G, int group) {
  const int r = blockIdx.x;
  for (int g = threadIdx.x; g < G; g += blockDim.x) {
    const size_t base = size_t(r) * cols + size_t(g) * group;
    float lo = 1e30f, hi = -1e30f;
    for (int i = 0; i < group; ++i) {
      const float v = __bfloat162float(src[base + i]);
      lo = fminf(lo, v); hi = fmaxf(hi, v);
    }
    // Measured: searching a clipping ratio to minimise squared error made this
    // WORSE (KL 1.8e-2 -> 5.9e-2 at g128). For lm_head a group's outliers are
    // the signal for rare tokens, so clipping them trades a small average error
    // for a large error exactly where it matters. Plain min/max it is.
    float best_s = (hi - lo) / 15.0f;
    if (!(best_s > 0.f)) best_s = 1e-8f;
    int best_z = max(0, min(15, __float2int_rn(-lo / best_s)));
    scale[size_t(r) * G + g] = __float2bfloat16(best_s);
    // zero points pack 8 rows per word along the OUTPUT axis
    atomicOr(&zp[(size_t(r) >> 3) * G + g], uint32_t(best_z) << ((r & 7) * 4));
    const float inv = 1.0f / best_s;
    for (int i = 0; i < group; i += 8) {
      uint32_t w = 0;
      for (int k = 0; k < 8; ++k) {
        int q = __float2int_rn(__bfloat162float(src[base + i + k]) * inv) + best_z;
        q = max(0, min(15, q));
        w |= uint32_t(q) << (4 * k);
      }
      packed[size_t(r) * (cols / 8) + (size_t(g) * group + i) / 8] = w;
    }
  }
}

void quantize_w4a16_raw(W4A16Weights& dst, const __nv_bfloat16* src,
                        int rows, int cols, int group) {
  const int G = cols / group;
  uint32_t *tp, *tz; __nv_bfloat16* ts;
  CKM(cudaMalloc(&tp, size_t(rows) * cols / 8 * 4));
  CKM(cudaMalloc(&ts, size_t(rows) * G * 2));
  CKM(cudaMalloc(&tz, size_t(rows) / 8 * G * 4));
  CKM(cudaMemset(tz, 0, size_t(rows) / 8 * G * 4));
  k_quant_i4<<<rows, 128>>>(tp, ts, tz, src, cols, G, group);
  CKM(cudaDeviceSynchronize());
  awq_alloc_fused(dst, rows, cols, group);
  awq_repack_into(dst, 0, tp, ts, tz, rows);
  CKM(cudaDeviceSynchronize());
  cudaFree(tp); cudaFree(ts); cudaFree(tz);
}

void quantize_to_w4a16(Model& m, W4A16Weights& dst, const __nv_bfloat16* src,
                       int rows, int cols, int group) {
  quantize_w4a16_raw(dst, src, rows, cols, group);
  m.owned.push_back(dst.qweight); m.owned.push_back(dst.scale); m.owned.push_back(dst.zp);
}

const char* human(size_t b, char* buf) {
  snprintf(buf, 32, "%7.3f GiB", double(b) / 1073741824.0);
  return buf;
}

}  // namespace

Model::~Model() {
  for (void* p : owned) cudaFree(p);
  owned.clear();
  if (cublas) cublasDestroy(cublas);
  gemv_scratch_free(gemv);
}

void model_load(Model& m, const std::string& dir, const LoadOptions& opt) {
  m.shape = ModelShape::from_file(dir + "/config.json");
  const ModelShape& S = m.shape;
  m.gdn.hidden = S.hidden_size;
  m.gdn.num_v_heads = S.linear_num_value_heads;
  m.gdn.num_k_heads = S.linear_num_key_heads;
  m.gdn.head_k = S.linear_key_head_dim;
  m.gdn.head_v = S.linear_value_head_dim;
  m.gdn.conv_k = S.linear_conv_kernel_dim;
  m.gdn.rms_eps = S.rms_norm_eps;
  m.attn.hidden = S.hidden_size;
  m.attn.num_q_heads = S.num_attention_heads;
  m.attn.num_kv_heads = S.num_key_value_heads;
  m.attn.head_dim = S.head_dim;
  m.attn.rotary_dim = S.rotary_dims;
  m.attn.rope_theta = float(S.rope_theta);
  m.attn.rms_eps = S.rms_norm_eps;
  // --max-context auto (opt.max_ctx == 0): size the KV cache from whatever the
  // device has left once everything else is resident. Until then use the model's
  // trained maximum, so any scratch sized from max_ctx is an upper bound.
  const bool auto_ctx = (opt.max_ctx <= 0);
  m.max_ctx = auto_ctx ? S.max_position_embeddings : opt.max_ctx;
  m.max_batch = opt.max_batch;
  m.lm_head_bits = opt.lm_head_bits;

  size_t free_b = 0, total_b = 0;
  CKM(cudaMemGetInfo(&free_b, &total_b));
  if (opt.verbose)
    printf("VRAM: %.2f GiB free of %.2f GiB\n", free_b / 1073741824.0, total_b / 1073741824.0);

  SafeTensors st;
  st.open_dir(dir);
  cublasCreate(&m.cublas);

  const int GROUP = S.quant_group_size;
  const std::string LP = "model.language_model.layers.";
  m.layers.resize(S.num_hidden_layers);

  size_t body_bytes = 0;
  for (int i = 0; i < S.num_hidden_layers; ++i) {
    LayerWeights& L = m.layers[i];
    const std::string p = LP + std::to_string(i) + ".";
    L.is_attn = S.layer_types[i] == LayerKind::FullAttention;
    L.attn_slot = S.attn_layer_index[i];
    L.input_ln = upload(m, st, p + "input_layernorm.weight");
    L.post_ln  = upload(m, st, p + "post_attention_layernorm.weight");
    // gate|up share an input, so they fuse into one GEMV
    fuse(m, st, L.mlp_gate_up, {p + "mlp.gate_proj", p + "mlp.up_proj"}, S.hidden_size, GROUP);
    fuse(m, st, L.mlp_down, {p + "mlp.down_proj"}, S.intermediate_size, GROUP);
    if (L.is_attn) {
      fuse(m, st, L.attn_qkv, {p + "self_attn.q_proj", p + "self_attn.k_proj",
                               p + "self_attn.v_proj"}, S.hidden_size, GROUP);
      fuse(m, st, L.attn_o, {p + "self_attn.o_proj"}, int(m.attn.o_proj_in()), GROUP);
      L.q_norm = upload(m, st, p + "self_attn.q_norm.weight");
      L.k_norm = upload(m, st, p + "self_attn.k_norm.weight");
      body_bytes += L.attn_qkv.total_bytes() + L.attn_o.total_bytes();
    } else {
      fuse(m, st, L.gdn_in_qkvz, {p + "linear_attn.in_proj_qkv", p + "linear_attn.in_proj_z"},
           S.hidden_size, GROUP);
      fuse(m, st, L.gdn_out, {p + "linear_attn.out_proj"}, int(m.gdn.val_dim()), GROUP);
      L.gdn_a = upload(m, st, p + "linear_attn.in_proj_a.weight");
      L.gdn_b = upload(m, st, p + "linear_attn.in_proj_b.weight");
      L.gdn_conv_w = upload(m, st, p + "linear_attn.conv1d.weight");
      L.gdn_A_log = upload(m, st, p + "linear_attn.A_log");
      L.gdn_dt_bias = upload(m, st, p + "linear_attn.dt_bias");
      L.gdn_norm = upload(m, st, p + "linear_attn.norm.weight");
      body_bytes += L.gdn_in_qkvz.total_bytes() + L.gdn_out.total_bytes();
    }
    body_bytes += L.mlp_gate_up.total_bytes() + L.mlp_down.total_bytes();
    if (opt.verbose && (i + 1) % 16 == 0) { printf("  loaded %d/%d layers\n", i + 1, S.num_hidden_layers); fflush(stdout); }
  }
  m.final_norm = upload(m, st, "model.language_model.norm.weight");

  // ---- embeddings and lm_head ------------------------------------------
  const auto& emb = st.get("model.language_model.embed_tokens.weight");
  if (opt.embed_host) {
    // Straight off the mapping into our own host buffer, in bf16. The embedding
    // never reaches the device: it is a row gather, not a matmul, so one row per
    // token over PCIe buys back 1.185 GiB with no accuracy cost at all.
    m.embed_host_bf = static_cast<__nv_bfloat16*>(malloc(emb.nbytes));
    if (!m.embed_host_bf) {
      fprintf(stderr, "embed_host: malloc of %zu bytes failed\n", emb.nbytes);
      abort();
    }
    memcpy(m.embed_host_bf, emb.data, emb.nbytes);
    CKM(cudaHostAlloc(reinterpret_cast<void**>(&m.embed_stage),
                      size_t(m.max_batch) * S.hidden_size * 2, cudaHostAllocDefault));
    m.embed_on_host = true;
    m.embed_quantized = false;
  } else {
    void* tmp = nullptr;
    CKM(cudaMalloc(&tmp, emb.nbytes));
    CKM(cudaMemcpy(tmp, emb.data, emb.nbytes, cudaMemcpyHostToDevice));
    if (opt.quantize_embed) {
      m.embed_q = static_cast<int8_t*>(dalloc(m, size_t(S.vocab_size) * S.hidden_size));
      m.embed_scale = static_cast<float*>(dalloc(m, size_t(S.vocab_size) * 4));
      quantize_embed_int8(m.embed_q, m.embed_scale,
                          static_cast<const __nv_bfloat16*>(tmp), S.vocab_size, S.hidden_size);
      CKM(cudaDeviceSynchronize());
      cudaFree(tmp);
    } else {
      m.embed_bf = static_cast<__nv_bfloat16*>(tmp);
      m.owned.push_back(tmp);
    }
    m.embed_quantized = opt.quantize_embed;
  }
  {
    const auto& lh = st.get("lm_head.weight");
    void* tmp = nullptr;
    CKM(cudaMalloc(&tmp, lh.nbytes));
    CKM(cudaMemcpy(tmp, lh.data, lh.nbytes, cudaMemcpyHostToDevice));
    if (opt.lm_head_bits == 4) {
      quantize_to_w4a16(m, m.lm_head_q, static_cast<const __nv_bfloat16*>(tmp),
                        S.vocab_size, S.hidden_size, opt.lm_head_group);
      cudaFree(tmp);
    } else if (opt.lm_head_bits == 8) {
      quantize_w8a16(m.lm_head_q8, static_cast<const __nv_bfloat16*>(tmp),
                     S.vocab_size, S.hidden_size, 128);
      CKM(cudaDeviceSynchronize());
      m.owned.push_back(m.lm_head_q8.qweight);
      m.owned.push_back(m.lm_head_q8.scale);
      cudaFree(tmp);
    } else {
      m.lm_head_bf16 = static_cast<__nv_bfloat16*>(tmp);
      m.owned.push_back(tmp);
    }
  }

  // ---- state ------------------------------------------------------------
  // The KV cache is allocated LAST, after every fixed cost is on the device, so
  // that --max-context auto can hand it whatever is actually left. Sizing it
  // first is what made an over-large context OOM inside a later allocation
  // instead of being refused at startup.
  m.gdn_state = static_cast<float*>(dalloc(m, size_t(S.gdn_state_elems()) * 4));
  m.gdn_conv  = static_cast<float*>(dalloc(m, size_t(S.gdn_conv_state_elems()) * 4));
  CKM(cudaMemset(m.gdn_state, 0, size_t(S.gdn_state_elems()) * 4));
  CKM(cudaMemset(m.gdn_conv, 0, size_t(S.gdn_conv_state_elems()) * 4));

  // ---- scratch ----------------------------------------------------------
  const int B = m.max_batch, H = S.hidden_size;
  const int qkv_w = int(m.attn.q_proj_out() + 2 * m.attn.kv_proj_out());
  const int gdn_w = int(m.gdn.conv_dim() + m.gdn.val_dim());
  const int widest = std::max({qkv_w, gdn_w, 2 * S.intermediate_size});
  m.h        = static_cast<__nv_bfloat16*>(dalloc(m, size_t(B) * H * 2));
  m.h2       = static_cast<__nv_bfloat16*>(dalloc(m, size_t(B) * H * 2));
  m.proj     = static_cast<__nv_bfloat16*>(dalloc(m, size_t(B) * widest * 2));
  m.mlp_tmp  = static_cast<__nv_bfloat16*>(dalloc(m, size_t(B) * S.intermediate_size * 2));
  m.q_buf    = static_cast<__nv_bfloat16*>(dalloc(m, size_t(B) * m.attn.o_proj_in() * 2));
  m.gate_buf = static_cast<__nv_bfloat16*>(dalloc(m, size_t(B) * m.attn.o_proj_in() * 2));
  m.attn_out = static_cast<__nv_bfloat16*>(dalloc(m, size_t(B) * m.attn.o_proj_in() * 2));
  m.gdn_qkv  = static_cast<__nv_bfloat16*>(dalloc(m, size_t(B) * m.gdn.conv_dim() * 2));
  m.gdn_core = static_cast<__nv_bfloat16*>(dalloc(m, size_t(B) * m.gdn.val_dim() * 2));
  m.gdn_ab   = static_cast<__nv_bfloat16*>(dalloc(m, size_t(B) * 2 * S.linear_num_value_heads * 2));
  m.gdn_g    = static_cast<float*>(dalloc(m, size_t(B) * S.linear_num_value_heads * 4));
  m.gdn_beta = static_cast<float*>(dalloc(m, size_t(B) * S.linear_num_value_heads * 4));
  m.cos_tab  = static_cast<float*>(dalloc(m, size_t(B) * S.rotary_dims / 2 * 4));
  m.sin_tab  = static_cast<float*>(dalloc(m, size_t(B) * S.rotary_dims / 2 * 4));
  m.logits   = static_cast<__nv_bfloat16*>(dalloc(m, size_t(S.vocab_size) * 2));
  m.pos_buf  = static_cast<int32_t*>(dalloc(m, size_t(B) * 4));
  m.pos_t    = static_cast<int32_t*>(dalloc(m, size_t(B) * 4));
  m.pos_h    = static_cast<int32_t*>(dalloc(m, size_t(B) * 4));
  m.pos_w    = static_cast<int32_t*>(dalloc(m, size_t(B) * 4));
  m.id_buf   = static_cast<int32_t*>(dalloc(m, size_t(B) * 4));
  // 256 partial indices + 256 partial values = 512 ints, then room for the
  // chosen id and slack. The previous 256*8 allocation was EXACTLY 512 ints and
  // callers wrote the result at +512, one past the end -- it happened to work
  // until an adjacent allocation moved.
  m.argmax_scratch = static_cast<int32_t*>(dalloc(m, 4096));
  m.graph_splits = attn_decode_splits(m.max_ctx);
  const int maxsp = m.graph_splits;
  m.attn_ws  = static_cast<float*>(dalloc(m, attn_decode_workspace_bytes(m.attn, maxsp)));
  m.kv_deq   = static_cast<__nv_bfloat16*>(dalloc(m, size_t(2) * 2048 * S.num_key_value_heads * S.head_dim * 2));
  m.prefill_scores = static_cast<float*>(dalloc(m, attn_prefill_scratch_bytes(m.attn, 2048, 128)));
  gemv_scratch_alloc(m.gemv, widest, std::max(S.vocab_size, widest), GROUP, 16);
  gemm_workspace_alloc(m.gemm, 256ull << 20);
  m.owned.push_back(m.gemm.wbuf);

  // ---- KV cache, sized against what is actually left --------------------
  const size_t kv_per_token = size_t(S.num_attn_layers) * S.num_key_value_heads *
                              S.head_dim;   // per side; k and v are separate
  const size_t margin = 512ull << 20;
  CKM(cudaMemGetInfo(&free_b, &total_b));
  if (auto_ctx) {
    const size_t avail = free_b > margin ? free_b - margin : 0;
    size_t fit = avail / (2 * kv_per_token);
    fit = (fit / 256) * 256;                       // whole graph buckets
    if (fit > size_t(S.max_position_embeddings)) fit = S.max_position_embeddings;
    if (fit < 1024) {
      fprintf(stderr, "\nFATAL: only %.0f MB free after the model and workspaces; "
                      "not enough for a usable KV cache.\n", free_b / 1048576.0);
      abort();
    }
    m.max_ctx = int(fit);
    if (opt.verbose)
      printf("\n--max-context auto -> %d tokens (%.2f GiB of KV, %.0f MB margin)\n",
             m.max_ctx, double(2 * kv_per_token * m.max_ctx) / (1 << 30),
             margin / 1048576.0);
  }
  const size_t kv_bytes = kv_per_token * m.max_ctx;
  m.k_cache = static_cast<uint8_t*>(dalloc(m, kv_bytes));
  m.v_cache = static_cast<uint8_t*>(dalloc(m, kv_bytes));
  // splits depend on the final context, and the eager and graph paths must agree
  m.graph_splits = attn_decode_splits(m.max_ctx);

  CKM(cudaMemGetInfo(&free_b, &total_b));
  if (opt.verbose) {
    char b1[32];
    printf("\n=== VRAM ===\n");
    printf("  body (int4 fused)     %s\n", human(body_bytes, b1));
    const size_t lmb = opt.lm_head_bits == 4 ? m.lm_head_q.total_bytes()
                     : opt.lm_head_bits == 8 ? m.lm_head_q8.total_bytes()
                     : size_t(S.vocab_size) * S.hidden_size * 2;
    printf("  lm_head               %s (%d-bit)\n", human(lmb, b1), opt.lm_head_bits);
    printf("  embed_tokens          %s %s\n",
           human(opt.quantize_embed ? size_t(S.vocab_size) * (S.hidden_size + 4)
                                    : size_t(S.vocab_size) * S.hidden_size * 2, b1),
           opt.quantize_embed ? "(INT8 rowwise)" : "(BF16)");
    printf("  KV cache @ %d ctx   %s (FP8, %lld KiB/token)\n", m.max_ctx,
           human(2 * kv_bytes, b1), (long long)(S.kv_bytes_per_token(1) / 1024));
    printf("  GDN state             %s\n", human(size_t(S.gdn_state_elems()) * 4, b1));
    printf("  free after load       %s\n", human(free_b, b1));
  }
  // The directive's hard contract: refuse to start rather than OOM at token
  // 40,000. 512 MB of margin.
  if (free_b < (512ull << 20)) {
    fprintf(stderr, "\nFATAL: only %.0f MB free after loading; --max-context %d does not fit "
                    "with a 512 MB margin.\n", free_b / 1048576.0, m.max_ctx);
    abort();
  }
}

// ================================================================= forward
namespace {

// Which linear path to use for T rows.
//   T == 1        : the GEMV, 807 GB/s
//   2 <= T <= 16  : the tensor-core W4A16 GEMM. Flat in T (0.21 ms at T=1 to
//                   0.22 at T=16 on the dominant tensor), which is what makes
//                   verifying a speculative block cost about one decode step.
//   T > 16        : dequantize to bf16 and hand it to cuBLAS, which is worth
//                   its extra traffic only once the arithmetic dominates.
enum class LinPath { Gemv, Mma, Cublas };
static LinPath lin_path(int T) {
  static int mode = -1;
  if (mode < 0) { const char* e = getenv("QWEN_LINPATH"); mode = e ? atoi(e) : 0; }
  if (T == 1) return LinPath::Gemv;
  if (mode == 1) return LinPath::Cublas;          // debug: no MMA
  // The tensor-core path takes at most 16 rows per launch, but it is weight-
  // stream bound and nearly flat in M, so chunking it by 16 costs a pass over
  // the weights per chunk and nothing else. cuBLAS, by contrast, dequantises
  // every weight to bf16 into a workspace first -- a fixed ~250 ms for the whole
  // model, measured at 12.55x a decode step at T=8. So cuBLAS only wins once T
  // is large enough to amortise that, which is far past 16. Getting this
  // crossover wrong made a 25-token warm prefill cost 286 ms.
  static int maxt = 0;
  if (!maxt) { const char* e = getenv("QWEN_MMA_MAX_T"); maxt = e ? atoi(e) : 128; }
  if (T <= maxt) return LinPath::Mma;
  return LinPath::Cublas;
}

static void linear(Model& m, __nv_bfloat16* out, const W4A16Weights& w,
                   const __nv_bfloat16* in, int T) {
  { static int on = -1;
    if (on < 0) { const char* e = getenv("QWEN_DEBUG_SYNC"); on = e && atoi(e); }
    if (on) fprintf(stderr, "  linear T=%d in=%d out=%d group=%d path=%d\n",
                    T, w.in_f, w.out_f, w.group_size, int(lin_path(T))); }
  switch (lin_path(T)) {
    case LinPath::Gemv:   gemv_w4a16(out, w, in, m.gemv, m.stream); break;
    case LinPath::Mma:
      for (int i = 0; i < T; i += 16) {
        const int n = std::min(16, T - i);
        gemm_mma_w4a16(out + size_t(i) * w.out_f, w, in + size_t(i) * w.in_f, n,
                       m.gemv, m.stream);
      }
      break;
    case LinPath::Cublas: gemm_w4a16(out, w, in, T, m.gemm, m.stream); break;
  }
}

void run_layer(Model& m, int li, int T, int position, bool prefill, bool dev_pos = false) {
  LayerWeights& L = m.layers[li];
  const ModelShape& S = m.shape;
  const int H = S.hidden_size;
  g_dbg_stream = m.stream;
  // Timing probe only: force the T=1 attention/GDN kernels at T>1. The result is
  // numerically wrong (attention sees one query row), but it isolates how much
  // of the T>=2 block cost is the prefill-shaped attention path.
  { static int f = -1;
    if (f < 0) { const char* e = getenv("QWEN_FORCE_DECODE"); f = e && atoi(e); }
    if (f) {
      static bool warned = false;
      if (!warned) { warned = true;
        fprintf(stderr, "*** QWEN_FORCE_DECODE=1: attention sees ONE query row at T>1. "
                        "Output is WRONG. Timing probe only. ***\n"); }
      prefill = false;
    } }

  // ---- token mixer ----
  rmsnorm(m.h2, m.h, L.input_ln, T, H, S.rms_norm_eps, m.stream);
  dsync("input_ln", li);
  if (L.is_attn) {
    const int qkv_w = int(m.attn.q_proj_out() + 2 * m.attn.kv_proj_out());
    linear(m, m.proj, L.attn_qkv, m.h2, T);
    dsync("attn_qkv", li);
    uint8_t* kcw = m.k_cache + size_t(L.attn_slot) * m.max_ctx * m.attn.kv_proj_out();
    uint8_t* vcw = m.v_cache + size_t(L.attn_slot) * m.max_ctx * m.attn.kv_proj_out();
    if (dev_pos)
      attn_prepare_dev(m.q_buf, m.gate_buf, kcw, vcw, m.proj, L.q_norm, L.k_norm,
                       m.cos_tab, m.sin_tab, m.d_step + 1, m.max_ctx, m.attn, m.stream);
    else
      attn_prepare(m.q_buf, m.gate_buf, kcw, vcw, m.proj, L.q_norm, L.k_norm,
                   m.cos_tab, m.sin_tab, T, position, m.max_ctx, m.attn, m.stream);
    dsync("attn_prepare", li);
    const uint8_t* kc = m.k_cache + size_t(L.attn_slot) * m.max_ctx * m.attn.kv_proj_out();
    const uint8_t* vc = m.v_cache + size_t(L.attn_slot) * m.max_ctx * m.attn.kv_proj_out();
    if (prefill) {
      attn_prefill(m.attn_out, m.q_buf, kc, vc, T, position + T, position, m.max_ctx,
                   m.attn, m.kv_deq, m.prefill_scores, m.cublas, m.stream);
    } else if (dev_pos) {
      attn_decode_dev(m.attn_out, m.q_buf, kc, vc, m.d_step + 2, m.max_ctx, m.attn,
                      m.attn_ws, m.graph_splits, m.stream);
    } else {
      // Fixed split count, chosen from max_ctx rather than the current context.
      // A context-dependent count changes the fp32 summation order of the
      // softmax combine, which makes decode non-reproducible across context
      // lengths AND makes the eager and graph paths disagree bit for bit. Empty
      // splits cost nothing: the kernel yields m=-FLT_MAX, l=0, which the
      // combine handles.
      attn_decode(m.attn_out, m.q_buf, kc, vc, position + 1, m.max_ctx, m.attn,
                  m.attn_ws, m.graph_splits, m.stream);
    }
    attn_output_gate(m.attn_out, m.gate_buf, T * int(m.attn.o_proj_in()), m.stream);
    dsync("attn_gate", li);
    linear(m, m.h2, L.attn_o, m.attn_out, T);
    dsync("attn_o", li);
  } else {
    linear(m, m.proj, L.gdn_in_qkvz, m.h2, T);
    dsync("gdn_in", li);
    // in_proj_a and in_proj_b are tiny bf16 tensors, so cuBLAS is fine
    const float one = 1.f, zero = 0.f;
    const int NV = S.linear_num_value_heads;
    cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N, NV, T, H, &one,
                 L.gdn_a, CUDA_R_16BF, H, m.h2, CUDA_R_16BF, H, &zero,
                 m.gdn_ab, CUDA_R_16BF, NV, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N, NV, T, H, &one,
                 L.gdn_b, CUDA_R_16BF, H, m.h2, CUDA_R_16BF, H, &zero,
                 m.gdn_ab + size_t(T) * NV, CUDA_R_16BF, NV,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    const int gi = li - L.attn_slot - 1;              // index among GDN layers
    const int gdn_idx = li - (li / 4);                // 3 of every 4 layers are GDN
    const int fused_w = int(m.gdn.conv_dim() + m.gdn.val_dim());
    gdn_conv(m.gdn_qkv, m.gdn_conv + size_t(gdn_idx) * m.gdn.conv_dim() * (m.gdn.conv_k - 1),
             // Always consume the conv state during decode: at position 0 the
             // state is zeroed, which is identical to not using it, and a
             // position-dependent bool would make the graph position-specific.
             m.proj, L.gdn_conv_w, int(m.gdn.conv_dim()), m.gdn.conv_k, T,
             dev_pos ? true : (position > 0), fused_w, m.stream);
    dsync("gdn_conv", li);
    gdn_gates(m.gdn_g, m.gdn_beta, m.gdn_ab, m.gdn_ab + size_t(T) * NV,
              L.gdn_A_log, L.gdn_dt_bias, T, NV, m.stream);
    dsync("gdn_gates", li);
    if (m.spec && m.spec->capturing && T <= m.spec->block) {
      // Capture BEFORE the scan advances the state: the replay needs the same
      // inputs the forward saw, and the projections must not be redone.
      const int CD = int(m.gdn.conv_dim());
      capture_pre_conv(m.spec->pre_conv + size_t(gdn_idx) * m.spec->block * CD,
                       m.proj, T, CD, fused_w, m.stream);
      CKM(cudaMemcpyAsync(m.spec->post_conv + size_t(gdn_idx) * m.spec->block * CD,
                          m.gdn_qkv, size_t(T) * CD * 2, cudaMemcpyDeviceToDevice, m.stream));
      CKM(cudaMemcpyAsync(m.spec->g + size_t(gdn_idx) * m.spec->block * NV,
                          m.gdn_g, size_t(T) * NV * 4, cudaMemcpyDeviceToDevice, m.stream));
      CKM(cudaMemcpyAsync(m.spec->beta + size_t(gdn_idx) * m.spec->block * NV,
                          m.gdn_beta, size_t(T) * NV * 4, cudaMemcpyDeviceToDevice, m.stream));
    }
    gdn_scan(m.gdn_core, m.gdn_state + size_t(gdn_idx) * NV * m.gdn.head_k * m.gdn.head_v,
             m.gdn_qkv, m.gdn_g, m.gdn_beta, m.gdn, T, m.stream);
    dsync("gdn_scan", li);
    // z is the second half of the fused in_proj output, so it is at column
    // conv_dim of a row that is conv_dim + val_dim wide.
    gdn_norm_gate(m.gdn_core, m.gdn_core, m.proj + m.gdn.conv_dim(), L.gdn_norm, T, NV,
                  m.gdn.head_v, S.rms_norm_eps, fused_w, m.stream);
    dsync("gdn_norm_gate", li);
    linear(m, m.h2, L.gdn_out, m.gdn_core, T);
    dsync("gdn_out", li);
    (void)gi;
  }
  dsync("mixer_residual", li);
  residual_add(m.h, m.h2, T * H, m.stream);

  // ---- mlp ----
  rmsnorm(m.h2, m.h, L.post_ln, T, H, S.rms_norm_eps, m.stream);
  dsync("post_ln", li);
  linear(m, m.proj, L.mlp_gate_up, m.h2, T);
  dsync("mlp_gate_up", li);
  swiglu(m.mlp_tmp, m.proj, T, S.intermediate_size, m.stream);
  dsync("swiglu", li);
  linear(m, m.h2, L.mlp_down, m.mlp_tmp, T);
  dsync("mlp_down", li);
  residual_add(m.h, m.h2, T * H, m.stream);
  dsync("mlp_residual", li);

  if (m.tap_enable && m.tap_of[li] >= 0) {
    const int stride = m.n_taps * H;
    const size_t n = size_t(T) * H;
    k_tap<<<(n + 255) / 256, 256, 0, m.stream>>>(m.taps, m.h, T, H, stride,
                                                 m.tap_of[li] * H);
  }
}

void head(Model& m, int T) {
  const ModelShape& S = m.shape;
  // logits only for the last position
  __nv_bfloat16* last = m.h + size_t(T - 1) * S.hidden_size;
  rmsnorm(m.h2, last, m.final_norm, 1, S.hidden_size, S.rms_norm_eps, m.stream);
  if (m.lm_head_bits == 4) {
    gemv_w4a16(m.logits, m.lm_head_q, m.h2, m.gemv, m.stream);
  } else if (m.lm_head_bits == 8) {
    gemv_w8a16(m.logits, m.lm_head_q8, m.h2, m.gemv, m.stream);
  } else {
    const float one = 1.f, zero = 0.f;
    cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N, S.vocab_size, 1, S.hidden_size,
                 &one, m.lm_head_bf16, CUDA_R_16BF, S.hidden_size,
                 m.h2, CUDA_R_16BF, S.hidden_size, &zero,
                 m.logits, CUDA_R_16BF, S.vocab_size,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
}

}  // namespace

void model_prefill_mm(Model& m, const int32_t* ids, int T, int position,
                      const int32_t* pt, const int32_t* ph, const int32_t* pw,
                      const EmbedSplice* splices, int n_splices) {
  const ModelShape& S = m.shape;
  // The activation buffers, m.id_buf and the embedding staging buffer are all
  // sized for max_batch. A caller chunking by anything wider silently walks off
  // the end of them.
  if (T > m.max_batch) {
    fprintf(stderr, "model_prefill: T=%d exceeds max_batch=%d\n", T, m.max_batch);
    abort();
  }
  CKM(cudaMemcpy(m.id_buf, ids, size_t(T) * 4, cudaMemcpyHostToDevice));
  CKM(cudaMemcpy(m.pos_t, pt, size_t(T) * 4, cudaMemcpyHostToDevice));
  CKM(cudaMemcpy(m.pos_h, ph, size_t(T) * 4, cudaMemcpyHostToDevice));
  CKM(cudaMemcpy(m.pos_w, pw, size_t(T) * 4, cudaMemcpyHostToDevice));

  if (m.embed_on_host)        embed_rows_host(m, ids, T);
  else if (m.embed_quantized) embed_int8(m.h, m.embed_q, m.embed_scale, m.id_buf, T, S.hidden_size, m.stream);
  else                        embed_bf16(m.h, m.embed_bf, m.id_buf, T, S.hidden_size, m.stream);

  // Image tokens: overwrite the placeholder embedding with the vision tower's
  // output. `deepstack_visual_indexes` is empty in this checkpoint, so this is
  // the only place visual features enter the language model.
  for (int i = 0; i < n_splices; ++i) {
    const EmbedSplice& sp = splices[i];
    if (sp.n_rows <= 0) continue;
    if (sp.dst_row < 0 || sp.dst_row + sp.n_rows > T) {
      fprintf(stderr, "model_prefill_mm: splice [%d,%d) outside chunk of %d\n",
              sp.dst_row, sp.dst_row + sp.n_rows, T);
      abort();
    }
    CKM(cudaMemcpyAsync(m.h + size_t(sp.dst_row) * S.hidden_size, sp.src,
                        size_t(sp.n_rows) * S.hidden_size * 2,
                        cudaMemcpyDeviceToDevice, m.stream));
  }

  if (m.dbg_hidden)
    CKM(cudaMemcpy(m.dbg_hidden, m.h, size_t(T) * S.hidden_size * 2, cudaMemcpyDeviceToDevice));
  rope_tables(m.cos_tab, m.sin_tab, m.pos_t, m.pos_h, m.pos_w, T,
              S.rotary_dims, float(S.rope_theta), m.stream);
  for (int i = 0; i < S.num_hidden_layers; ++i) {
    run_layer(m, i, T, position, T > 1);
    if (m.dbg_hidden)
      CKM(cudaMemcpy(m.dbg_hidden + size_t(i + 1) * m.max_batch * S.hidden_size, m.h,
                     size_t(T) * S.hidden_size * 2, cudaMemcpyDeviceToDevice));
  }
  head(m, T);
  m.ctx_len = position + T;
}

void model_prefill(Model& m, const int32_t* ids, int T, int position) {
  std::vector<int32_t> pos(T);
  for (int i = 0; i < T; ++i) pos[i] = position + i + m.mrope_delta;
  model_prefill_mm(m, ids, T, position, pos.data(), pos.data(), pos.data(), nullptr, 0);
}

int model_graph_bucket(const Model& m, int ctx) {
  for (int i = 0; i < m.n_graphs; ++i) if (ctx <= m.graph_ctx[i]) return i;
  return m.n_graphs - 1;
}

void model_graph_capture(Model& m) {
  const ModelShape& S = m.shape;
  if (!m.d_step) {
    m.d_step = static_cast<int32_t*>(dalloc(m, 16));
    CKM(cudaHostAlloc(&m.h_step, 16, cudaHostAllocDefault));
  }
  CKM(cudaStreamCreate(&m.capture_stream));

  const int cand[] = {8192, 32768, 131072};
  m.n_graphs = 0;
  for (int c : cand) {
    const int b = std::min(c, m.max_ctx);
    if (m.n_graphs && m.graph_ctx[m.n_graphs - 1] >= b) continue;
    m.graph_ctx[m.n_graphs] = b;
    m.graph_splits_of[m.n_graphs] = attn_decode_splits(b);
    ++m.n_graphs;
    if (b >= m.max_ctx) break;
  }

  cudaStream_t saved = m.stream;
  m.stream = m.capture_stream;
  cublasSetStream(m.cublas, m.capture_stream);

  auto body = [&]() {
    // With a host-resident table the row is DMA'd into m.h before the graph is
    // launched: the gather index is host data and cannot be a graph node. The
    // graph therefore starts at the rope tables, and m.h is a graph input.
    if (m.embed_on_host) {
      /* filled by embed_rows_host() outside the capture */
    } else if (m.embed_quantized) {
      embed_int8(m.h, m.embed_q, m.embed_scale, m.d_step, 1, S.hidden_size, m.stream);
    } else {
      embed_bf16(m.h, m.embed_bf, m.d_step, 1, S.hidden_size, m.stream);
    }
    rope_tables_dev(m.cos_tab, m.sin_tab, m.d_step + 3, S.rotary_dims,
                    float(S.rope_theta), m.stream);
    for (int i = 0; i < S.num_hidden_layers; ++i)
      run_layer(m, i, 1, 0, false, /*dev_pos=*/true);
    head(m, 1);
  };

  // Warm up OUTSIDE capture: the one-shot cudaFuncSetAttribute opt-ins and any
  // lazy cuBLAS workspace allocation must not happen during capture, or the
  // graph records an allocation it cannot replay.
  m.h_step[0] = 0; m.h_step[1] = 0; m.h_step[2] = 1; m.h_step[3] = 0;
  CKM(cudaMemcpy(m.d_step, m.h_step, 16, cudaMemcpyHostToDevice));
  for (int g = 0; g < m.n_graphs; ++g) {
    m.graph_splits = m.graph_splits_of[g];
    // Warm up OUTSIDE capture: the one-shot cudaFuncSetAttribute opt-ins and any
    // lazy cuBLAS workspace allocation must not happen during capture, or the
    // graph records an allocation it cannot replay.
    body();
    CKM(cudaStreamSynchronize(m.capture_stream));
    CKM(cudaStreamBeginCapture(m.capture_stream, cudaStreamCaptureModeThreadLocal));
    body();
    CKM(cudaStreamEndCapture(m.capture_stream, &m.graph[g]));
    CKM(cudaGraphInstantiate(&m.graph_exec[g], m.graph[g], nullptr, nullptr, 0));
  }

  m.stream = saved;
  cublasSetStream(m.cublas, saved);
  size_t nodes = 0;
  cudaGraphGetNodes(m.graph[0], nullptr, &nodes);
  printf("CUDA graphs captured: %d buckets, %zu nodes each (ctx<=", m.n_graphs, nodes);
  for (int g = 0; g < m.n_graphs; ++g)
    printf("%d:s%d%s", m.graph_ctx[g], m.graph_splits_of[g], g + 1 < m.n_graphs ? ", " : ")\n");
}

void model_forward_all_logits(Model& m, const int32_t* ids, int T, int position,
                              __nv_bfloat16* logits_out) {
  const ModelShape& S = m.shape;
  CKM(cudaMemcpyAsync(m.id_buf, ids, size_t(T) * 4, cudaMemcpyHostToDevice, m.stream));
  std::vector<int32_t> pos(T);
  for (int i = 0; i < T; ++i) pos[i] = position + i;
  CKM(cudaMemcpyAsync(m.pos_buf, pos.data(), size_t(T) * 4, cudaMemcpyHostToDevice, m.stream));
  if (m.embed_on_host)        embed_rows_host(m, ids, T);
  else if (m.embed_quantized) embed_int8(m.h, m.embed_q, m.embed_scale, m.id_buf, T, S.hidden_size, m.stream);
  else                        embed_bf16(m.h, m.embed_bf, m.id_buf, T, S.hidden_size, m.stream);
  rope_tables(m.cos_tab, m.sin_tab, m.pos_buf, m.pos_buf, m.pos_buf, T,
              S.rotary_dims, float(S.rope_theta), m.stream);
  for (int i = 0; i < S.num_hidden_layers; ++i) run_layer(m, i, T, position, T > 1);
  rmsnorm(m.h2, m.h, m.final_norm, T, S.hidden_size, S.rms_norm_eps, m.stream);
  if (m.lm_head_bits == 4)      gemm_mma_w4a16(logits_out, m.lm_head_q, m.h2, T, m.gemv, m.stream);
  else if (m.lm_head_bits == 8) gemm_small_w8a16(logits_out, m.lm_head_q8, m.h2, T, m.gemv, m.stream);
  else {
    const float one = 1.f, zero = 0.f;
    cublasSetStream(m.cublas, m.stream);
    cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N, S.vocab_size, T, S.hidden_size,
                 &one, m.lm_head_bf16, CUDA_R_16BF, S.hidden_size,
                 m.h2, CUDA_R_16BF, S.hidden_size, &zero,
                 logits_out, CUDA_R_16BF, S.vocab_size,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
  m.ctx_len = position + T;
}

void model_decode(Model& m, int32_t id, int position) {
  if (m.use_graph && m.n_graphs) {
    const int g = model_graph_bucket(m, position + 1);
    m.graph_splits = m.graph_splits_of[g];
    // [1] is the KV slot, [3] is the rope position. They are the same thing
    // until an image is in the context.
    m.h_step[0] = id; m.h_step[1] = position; m.h_step[2] = position + 1;
    m.h_step[3] = position + m.mrope_delta;
    CKM(cudaMemcpy(m.d_step, m.h_step, 16, cudaMemcpyHostToDevice));
    if (m.embed_on_host) embed_rows_host(m, &id, 1);
    CKM(cudaGraphLaunch(m.graph_exec[g], m.stream));
    m.ctx_len = position + 1;
    return;
  }
  // Eager path must pick the SAME split count the graph would, or the two are
  // not bitwise identical (gate_graph).
  if (m.n_graphs) m.graph_splits = m.graph_splits_of[model_graph_bucket(m, position + 1)];
  model_prefill(m, &id, 1, position);
}

std::vector<int32_t> model_generate_greedy(Model& m, const std::vector<int32_t>& prompt,
                                           int max_new, int eos_id) {
  std::vector<int32_t> out;
  model_prefill(m, prompt.data(), int(prompt.size()), 0);
  int32_t* d_id = m.argmax_scratch + 512;
  for (int i = 0; i < max_new; ++i) {
    argmax(d_id, m.logits, m.shape.vocab_size, m.argmax_scratch);
    int32_t tok = 0;
    CKM(cudaMemcpy(&tok, d_id, 4, cudaMemcpyDeviceToHost));
    out.push_back(tok);
    if (m.shape.is_stop_token(tok)) break;
    if (i + 1 < max_new) model_decode(m, tok, int(prompt.size()) + i);
  }
  return out;
}

// Public entry to the INT4 quantiser, for callers that own their own memory
// (the drafter). Ownership of dst.{qweight,scale,zp} passes to the caller.
void quantize_w4a16(W4A16Weights& dst, const __nv_bfloat16* src, int rows, int cols,
                    int group) {
  quantize_w4a16_raw(dst, src, rows, cols, group);
}

void model_apply_head(Model& m, __nv_bfloat16* out, const __nv_bfloat16* x, int T) {
  const ModelShape& S = m.shape;
  if (m.lm_head_bits == 4)      gemm_mma_w4a16(out, m.lm_head_q, x, T, m.gemv, m.stream);
  else if (m.lm_head_bits == 8) gemm_small_w8a16(out, m.lm_head_q8, x, T, m.gemv, m.stream);
  else {
    const float one = 1.f, zero = 0.f;
    cublasSetStream(m.cublas, m.stream);
    CKM(cudaGetLastError());
    cublasGemmEx(m.cublas, CUBLAS_OP_T, CUBLAS_OP_N, S.vocab_size, T, S.hidden_size,
                 &one, m.lm_head_bf16, CUDA_R_16BF, S.hidden_size, x, CUDA_R_16BF,
                 S.hidden_size, &zero, out, CUDA_R_16BF, S.vocab_size,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  }
}

void embed_rows_host(Model& m, const int32_t* ids, int T) {
  const int H = m.shape.hidden_size;
  if (T > m.max_batch) { fprintf(stderr, "embed_rows_host: T %d > max_batch\n", T); abort(); }
  for (int i = 0; i < T; ++i) {
    const int32_t id = ids[i];
    if (id < 0 || id >= m.shape.vocab_size) {
      fprintf(stderr, "embed_rows_host: token %d out of range\n", id);
      abort();
    }
    memcpy(m.embed_stage + size_t(i) * H, m.embed_host_bf + size_t(id) * H, size_t(H) * 2);
  }
  CKM(cudaMemcpyAsync(m.h, m.embed_stage, size_t(T) * H * 2, cudaMemcpyHostToDevice, m.stream));
}

void model_enable_taps(Model& m, const std::vector<int>& layer_ids) {
  const ModelShape& S = m.shape;
  m.tap_of.assign(S.num_hidden_layers, -1);
  for (size_t i = 0; i < layer_ids.size(); ++i) {
    const int li = layer_ids[i];
    if (li < 0 || li >= S.num_hidden_layers) {
      fprintf(stderr, "taps: layer %d out of range\n", li);
      abort();
    }
    m.tap_of[li] = int(i);
  }
  m.n_taps = int(layer_ids.size());
  if (!m.taps) {
    const size_t bytes = size_t(m.max_batch) * m.n_taps * S.hidden_size * 2;
    m.taps = static_cast<__nv_bfloat16*>(dalloc(m, bytes));
  }
  m.tap_enable = true;
}

void model_disable_taps(Model& m) { m.tap_enable = false; }

void dbg_profile_report(const char* tag) {
  if (!g_dbg_n) return;
  double tot = 0;
  for (int i = 0; i < g_dbg_n; ++i) tot += g_dbg[i].ms;
  printf("\n--- stage profile [%s]  total %.2f ms ---\n", tag, tot);
  for (int i = 0; i < g_dbg_n; ++i)
    printf("  %-14s %9.3f ms  %5ld calls  %7.3f ms/call  %5.1f%%\n",
           g_dbg[i].name, g_dbg[i].ms, g_dbg[i].n,
           g_dbg[i].n ? g_dbg[i].ms / g_dbg[i].n : 0.0, 100.0 * g_dbg[i].ms / tot);
  for (int i = 0; i < g_dbg_n; ++i) { g_dbg[i].ms = 0; g_dbg[i].n = 0; }
  g_dbg_last = 0;
}

}  // namespace qwen
