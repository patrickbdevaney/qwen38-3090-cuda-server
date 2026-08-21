#include "dflash.h"
#include "../loader/safetensors.h"
#include "../kernels/elementwise.cuh"
#include "../model/model.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <fstream>
#include "../../third_party/json.hpp"

namespace qwen {
namespace {

#define CKD(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); abort(); } } while(0)
#define CBD(x) do { cublasStatus_t st_=(x); if(st_!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS %s:%d status %d\n",__FILE__,__LINE__,int(st_)); abort(); } } while(0)

// ---------------------------------------------------------------- norms
//
// Qwen3RMSNorm, which is NOT the target model's Qwen3_5RMSNorm. Two differences
// and both matter: there is no "1.0 +" on the weight, and the reference writes
//     return self.weight * hidden_states.to(input_dtype)
// so the normalised value is ROUNDED TO BF16 BEFORE the weight multiply. Doing
// the multiply in fp32 and rounding once at the end is a different function.
__global__ void k_rmsnorm_plain(__nv_bfloat16* __restrict__ out,
                                const __nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ w,
                                int dim, float eps) {
  extern __shared__ float red[];
  const __nv_bfloat16* row = x + size_t(blockIdx.x) * dim;
  __nv_bfloat16* dst = out + size_t(blockIdx.x) * dim;
  float acc = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = __bfloat162float(row[i]);
    acc += v * v;
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = rsqrtf(red[0] / float(dim) + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    dst[i] = __hmul(__float2bfloat16(__bfloat162float(row[i]) * inv), w[i]);
}

// Per-head RMSNorm over head_dim, applied in place to a [rows, heads, hd] view.
__global__ void k_head_norm(__nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ w,
                            int heads, int hd, float eps) {
  const size_t base = (size_t(blockIdx.x) * heads + blockIdx.y) * hd;
  extern __shared__ float red[];
  float acc = 0.f;
  for (int i = threadIdx.x; i < hd; i += blockDim.x) {
    const float v = __bfloat162float(x[base + i]);
    acc += v * v;
  }
  red[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = rsqrtf(red[0] / float(hd) + eps);
  for (int i = threadIdx.x; i < hd; i += blockDim.x)
    x[base + i] = __hmul(__float2bfloat16(__bfloat162float(x[base + i]) * inv), w[i]);
}

// ---------------------------------------------------------------- rope
//
// inv_freq in DOUBLE, rounded once to float. __powf is accurate to ~1e-4
// relative, which at position 131071 is 13 radians of phase error -- and it
// looks perfect at position 0, so a short benchmark will not catch it.
__global__ void k_rope_tables(float* __restrict__ cs, float* __restrict__ sn,
                              const int32_t* __restrict__ pos, int rows, int half,
                              float theta) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * half) return;
  const int r = i / half, d = i % half;
  const double inv = 1.0 / pow(double(theta), (2.0 * d) / double(2 * half));
  const float f = float(pos[r]) * float(inv);
  cs[i] = cosf(f);
  sn[i] = sinf(f);
}

// HF rotate_half: the halves are SPLIT, not interleaved.
//   i <  half:  out[i] = q[i]*cos[i] - q[i+half]*sin[i]
//   i >= half:  out[i] = q[i]*cos[i-half] + q[i-half]*sin[i-half]
__global__ void k_rope_half(__nv_bfloat16* __restrict__ x,
                            const float* __restrict__ cs, const float* __restrict__ sn,
                            int heads, int hd, int half) {
  const int r = blockIdx.x, h = blockIdx.y;
  const size_t base = (size_t(r) * heads + h) * hd;
  const float* c = cs + size_t(r) * half;
  const float* s = sn + size_t(r) * half;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    const float a = __bfloat162float(x[base + i]);
    const float b = __bfloat162float(x[base + i + half]);
    x[base + i]        = __float2bfloat16(a * c[i] - b * s[i]);
    x[base + i + half] = __float2bfloat16(b * c[i] + a * s[i]);
  }
}

// ---------------------------------------------------------------- dynamic conv
//
// out[t][c] = sum_off (base[off][c] + dyn[t][which][off][g]) * in[t-off][c]
// with g = c / group_size, causal in t (positions before 0 contribute nothing).
//
// base_kernel ships as [2, kernel_size, hidden]. THE LEADING 2 IS NOT A TAP: it
// is {prepare, finish}, selected by `which`. Reading it as a third tap silently
// produces a plausible-looking wrong answer.
__global__ void k_dyn_conv(__nv_bfloat16* __restrict__ out,
                           const __nv_bfloat16* __restrict__ in,
                           const __nv_bfloat16* __restrict__ dyn,
                           const __nv_bfloat16* __restrict__ base,
                           int T, int hidden, int group, int taps, int groups,
                           int which) {
  // `base` must ALREADY point at base_kernel[which]; `which` here only selects
  // the dynamic half. Passing base_kernel and relying on `which` for both was a
  // bug that produced plausible-looking wrong activations.
  const int t = blockIdx.x;
  for (int c = threadIdx.x + blockIdx.y * blockDim.x; c < hidden;
       c += blockDim.x * gridDim.y) {
    const int g = c / group;
    float acc = 0.f;
    for (int off = 0; off < taps; ++off) {
      const int src = t - off;
      if (src < 0) continue;
      const float b = __bfloat162float(base[size_t(off) * hidden + c]);
      const float d = __bfloat162float(
          dyn[(size_t(t) * 2 + which) * taps * groups + size_t(off) * groups + g]);
      acc += (b + d) * __bfloat162float(in[size_t(src) * hidden + c]);
    }
    out[size_t(t) * hidden + c] = __float2bfloat16(acc);
  }
}

// ---------------------------------------------------------------- attention
//
// q_len is the block (8), kv_len is context + block. One warp per (head, query
// row); each lane owns head_dim/32 dims and the dot is a butterfly. The whole
// drafter is ~1.4 ms, of which this is a small slice, so it is written for
// clarity rather than for the last 10%.
__global__ void k_attn_draft(__nv_bfloat16* __restrict__ out,
                             const __nv_bfloat16* __restrict__ q,
                             const __nv_bfloat16* __restrict__ k,
                             const __nv_bfloat16* __restrict__ v,
                             int q_len, int kv_len, int nq, int nkv, int hd,
                             float scale, int window, int q_pos0, int k_pos0,
                             bool causal) {
  const int h = blockIdx.x;                       // query head
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int qr = warp;                            // one warp per query row
  if (qr >= q_len) return;
  const int kvh = h / (nq / nkv);
  const int dpl = hd / 32;                        // dims per lane
  const int d0 = lane * dpl;

  float qr_[8], acc[8];
  for (int i = 0; i < dpl; ++i) {
    qr_[i] = __bfloat162float(q[(size_t(qr) * nq + h) * hd + d0 + i]);
    acc[i] = 0.f;
  }
  float m = -INFINITY, l = 0.f;
  const int qpos = q_pos0 + qr;

  for (int p = 0; p < kv_len; ++p) {
    const int kpos = k_pos0 + p;
    // NOT CAUSAL. config.json sets "is_causal": false, and the reference then
    // masks by |qpos - kpos| < sliding_window in BOTH directions. That is what
    // makes this block diffusion rather than an ordinary draft model: all
    // block_size noise positions see each other, so one forward emits the whole
    // block instead of eight autoregressive steps. Reading the layer_type
    // ("sliding_attention") as implying causality gives a plausible-looking but
    // completely wrong answer -- measured 0.56 relative error at layer 0.
    if (causal && kpos > qpos) break;
    if (window > 0) {
      if (qpos - kpos >= window) continue;
      if (!causal && kpos - qpos >= window) continue;
    }
    float s = 0.f;
    for (int i = 0; i < dpl; ++i)
      s += qr_[i] * __bfloat162float(k[(size_t(p) * nkv + kvh) * hd + d0 + i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    s *= scale;
    const float mn = fmaxf(m, s);
    const float corr = __expf(m - mn), pw = __expf(s - mn);
    l = l * corr + pw;
    for (int i = 0; i < dpl; ++i)
      acc[i] = fmaf(acc[i], corr,
                    pw * __bfloat162float(v[(size_t(p) * nkv + kvh) * hd + d0 + i]));
    m = mn;
  }
  const float inv = l > 0.f ? 1.f / l : 0.f;
  for (int i = 0; i < dpl; ++i)
    out[(size_t(qr) * nq + h) * hd + d0 + i] = __float2bfloat16(acc[i] * inv);
}

// ---------------------------------------------------------------- top-k
//
// 16 passes over the row, each taking the largest (value, index) pair strictly
// below the previous one. No mutation and no scratch copy of the row; ties break
// on the smaller index, deterministically.
__global__ void k_topk(float* __restrict__ vals, int32_t* __restrict__ idx,
                       const __nv_bfloat16* __restrict__ logits, int vocab, int K) {
  __shared__ float sv[256];
  __shared__ int si[256];
  const __nv_bfloat16* row = logits + size_t(blockIdx.x) * vocab;
  float prev_v = INFINITY;
  int prev_i = -1;
  for (int p = 0; p < K; ++p) {
    float bv = -INFINITY;
    int bi = -1;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
      const float v = __bfloat162float(row[i]);
      // strictly below the previous pick in (value, -index) order
      if (v > prev_v || (v == prev_v && i <= prev_i)) continue;
      if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
    }
    sv[threadIdx.x] = bv; si[threadIdx.x] = bi;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        const float ov = sv[threadIdx.x + s];
        const int oi = si[threadIdx.x + s];
        if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
          sv[threadIdx.x] = ov; si[threadIdx.x] = oi;
        }
      }
      __syncthreads();
    }
    prev_v = sv[0]; prev_i = si[0];
    if (threadIdx.x == 0) {
      vals[size_t(blockIdx.x) * K + p] = prev_v;
      idx[size_t(blockIdx.x) * K + p] = prev_i;
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------- selector
//
// First-order Markov chain over the top-k lattice, greedy left to right:
//   score[c] = unary[pos][c] + sum_r pred[prev][r] * hproj[pos][r] * succ[cand[c]][r]
// The chain is sequential in pos, so this is one block that loops. rank is 256
// and top_k is 16, i.e. 4096 MACs per position -- the whole point of the
// selector is that it replaces a second 248320-wide projection.
__global__ void k_selector(int32_t* __restrict__ path,
                           const __nv_bfloat16* __restrict__ hproj,
                           const float* __restrict__ unary,
                           const int32_t* __restrict__ cand,
                           const __nv_bfloat16* __restrict__ pred_cb,
                           const __nv_bfloat16* __restrict__ succ_cb,
                           int n_pos, int K, int rank, int32_t anchor) {
  extern __shared__ float sh[];
  float* t = sh;                       // [rank]
  float* red = sh + rank;              // [blockDim.x]
  int32_t prev = anchor;
  for (int pos = 0; pos < n_pos; ++pos) {
    for (int r = threadIdx.x; r < rank; r += blockDim.x)
      t[r] = __bfloat162float(pred_cb[size_t(prev) * rank + r]) *
             __bfloat162float(hproj[size_t(pos) * rank + r]);
    __syncthreads();
    float best = -INFINITY;
    int besti = 0;
    for (int c = 0; c < K; ++c) {
      const int32_t tok = cand[size_t(pos) * K + c];
      float acc = 0.f;
      for (int r = threadIdx.x; r < rank; r += blockDim.x)
        acc += t[r] * __bfloat162float(succ_cb[size_t(tok) * rank + r]);
      red[threadIdx.x] = acc;
      __syncthreads();
      for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
      }
      const float score = unary[size_t(pos) * K + c] + red[0];
      if (score > best) { best = score; besti = c; }
      __syncthreads();
    }
    prev = cand[size_t(pos) * K + besti];
    if (threadIdx.x == 0) path[pos] = prev;
    __syncthreads();
  }
}

__global__ void k_scale(__nv_bfloat16* x, int n, float s) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = __float2bfloat16(__bfloat162float(x[i]) * s);
}

// ---------------------------------------------------------------- helpers
void* dmalloc(DraftModel& d, size_t bytes) {
  void* p = nullptr;
  CKD(cudaMalloc(&p, bytes));
  d.owned.push_back(p);
  d.bytes += bytes;
  return p;
}

// Copy a bf16 tensor from the mapping straight to device. No host staging: the
// drafter is 3.85 GB and a bounce buffer would be a second copy of it.
__nv_bfloat16* upload(DraftModel& d, const SafeTensors& st, const std::string& name,
                      int64_t expect_elems) {
  const TensorView& t = st.get(name);
  if (t.dtype != Dtype::BF16)
    { fprintf(stderr, "dflash: %s is %s, expected BF16\n", name.c_str(), dtype_name(t.dtype)); abort(); }
  if (expect_elems > 0 && t.numel() != expect_elems) {
    fprintf(stderr, "dflash: %s has %lld elements, expected %lld\n",
            name.c_str(), (long long)t.numel(), (long long)expect_elems);
    abort();
  }
  auto* p = static_cast<__nv_bfloat16*>(dmalloc(d, t.nbytes));
  CKD(cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice));
  return p;
}

// Concatenate several [out_i, in] tensors along the output dimension.
__nv_bfloat16* upload_fused(DraftModel& d, const SafeTensors& st,
                            const std::vector<std::string>& names, int in_f) {
  size_t total = 0;
  for (const auto& n : names) total += st.get(n).nbytes;
  auto* p = static_cast<__nv_bfloat16*>(dmalloc(d, total));
  size_t off = 0;
  for (const auto& n : names) {
    const TensorView& t = st.get(n);
    if (t.shape.size() != 2 || t.shape[1] != in_f) {
      fprintf(stderr, "dflash: %s shape mismatch (in_f %d)\n", n.c_str(), in_f);
      abort();
    }
    CKD(cudaMemcpy(reinterpret_cast<uint8_t*>(p) + off, t.data, t.nbytes,
                   cudaMemcpyHostToDevice));
    off += t.nbytes;
  }
  return p;
}

// Quantise a bf16 tensor already on the device into W4A16 and release the bf16
// copy. The drafter is 3.70 GiB in bf16 and 1.19 GiB at INT4, and the drafter's
// weights only affect draft QUALITY -- every drafted token is verified by the
// target -- so this trades a little acceptance for 2.5 GiB and a 3x cheaper
// draft step.
void quant_release(DraftModel& d, __nv_bfloat16*& src, W4A16Weights& dst,
                   int rows, int cols, int group) {
  if (rows % 32 || cols % group) {
    fprintf(stderr, "dflash quant: %dx%d not compatible with group %d\n", rows, cols, group);
    abort();
  }
  quantize_w4a16(dst, src, rows, cols, group);
  d.owned.push_back(dst.qweight);
  d.owned.push_back(dst.scale);
  d.owned.push_back(dst.zp);
  d.bytes += dst.total_bytes();
  // drop the bf16 original from the owned list and free it
  auto it = std::find(d.owned.begin(), d.owned.end(), static_cast<void*>(src));
  if (it != d.owned.end()) d.owned.erase(it);
  d.bytes -= size_t(rows) * cols * 2;
  cudaFree(src);
  src = nullptr;
}

}  // namespace

// y[M, out] = x[M, in] @ W^T, W row-major [out, in].
static void dlinear(DraftModel& d, __nv_bfloat16* y, const __nv_bfloat16* W,
                    const W4A16Weights& Wq, const __nv_bfloat16* x,
                    int out_f, int in_f, int M) {
  if (d.quantized && Wq.qweight) {
    if (M == 1) gemv_w4a16(y, Wq, x, d.gemv, d.stream);
    else if (M <= 16) gemm_mma_w4a16(y, Wq, x, M, d.gemv, d.stream);
    else {
      // The context push can be hundreds of rows; chunk it through the block
      // path rather than adding a second GEMM shape.
      for (int i = 0; i < M; i += 16) {
        const int n = std::min(16, M - i);
        gemm_mma_w4a16(y + size_t(i) * out_f, Wq, x + size_t(i) * in_f, n, d.gemv,
                       d.stream);
      }
    }
    return;
  }
  const float one = 1.f, zero = 0.f;
  CBD(cublasGemmEx(d.cublas, CUBLAS_OP_T, CUBLAS_OP_N, out_f, M, in_f, &one,
                   W, CUDA_R_16BF, in_f, x, CUDA_R_16BF, in_f, &zero,
                   y, CUDA_R_16BF, out_f, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void rmsnorm_plain(DraftModel& d, __nv_bfloat16* out, const __nv_bfloat16* x,
                          const __nv_bfloat16* w, int rows, int dim) {
  const int th = 256;
  k_rmsnorm_plain<<<rows, th, th * sizeof(float), d.stream>>>(out, x, w, dim, d.sh.rms_eps);
}

void draft_load(DraftModel& d, const std::string& dir, const DraftLoadOptions& opt) {
  using json = nlohmann::json;
  json cfg;
  { std::ifstream f(dir + "/config.json");
    if (!f) { fprintf(stderr, "dflash: no config.json in %s\n", dir.c_str()); abort(); }
    f >> cfg; }
  const json& dc = cfg.at("dflash_config");

  DFlashShape& s = d.sh;
  s.hidden = cfg.at("hidden_size");
  s.n_layers = cfg.at("num_hidden_layers");
  s.vocab = cfg.at("vocab_size");
  s.n_q_heads = cfg.at("num_attention_heads");
  s.n_kv_heads = cfg.at("num_key_value_heads");
  s.head_dim = cfg.at("head_dim");
  s.sliding_window = cfg.value("use_sliding_window", true) ? int(cfg.at("sliding_window")) : 0;
  s.rms_eps = cfg.at("rms_norm_eps");
  s.rope_theta = cfg.at("rope_parameters").at("rope_theta");
  s.block_size = dc.at("block_size");
  s.mask_token_id = dc.at("mask_token_id");
  s.top_k = dc.at("selector_top_k");
  s.rank = dc.at("selector_rank");
  s.conv_k = dc.at("conv_kernel_size");
  s.conv_group = dc.at("conv_group_size");
  s.embed_scale = dc.value("input_embedding_scale", 1.0f);
  // "is_causal" is a TOP-LEVEL config key, and this checkpoint sets it false.
  // The reference only falls back to layer_type == "sliding_attention" when the
  // key is absent entirely.
  s.is_causal = cfg.contains("is_causal") && !cfg["is_causal"].is_null()
                    ? cfg["is_causal"].get<bool>()
                    : (cfg.contains("layer_types") &&
                       cfg["layer_types"][0] == "sliding_attention");
  for (auto& v : dc.at("target_layer_ids")) s.target_layer_ids.push_back(v.get<int>());
  s.n_taps = int(s.target_layer_ids.size());
  d.inter = cfg.at("intermediate_size");
  d.quantized = opt.quantize;

  if (s.head_dim % 32 != 0 || s.head_dim > 256) {
    fprintf(stderr, "dflash: head_dim %d unsupported (needs %%32 and <=256)\n", s.head_dim);
    abort();
  }

  SafeTensors st;
  st.open_dir(dir);

  d.fc = upload(d, st, "fc.weight", int64_t(s.hidden) * s.n_taps * s.hidden);
  d.hidden_norm = upload(d, st, "hidden_norm.weight", s.hidden);
  d.final_norm = upload(d, st, "norm.weight", s.hidden);
  d.pred_cb = upload(d, st, "candidate_selector.predecessor_codebook",
                     int64_t(s.vocab) * s.rank);
  d.succ_cb = upload(d, st, "candidate_selector.successor_codebook",
                     int64_t(s.vocab) * s.rank);
  d.hproj = upload(d, st, "candidate_selector.hidden_projection.weight",
                   int64_t(s.rank) * s.hidden);

  d.layers.resize(s.n_layers);
  for (int i = 0; i < s.n_layers; ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    DraftLayer& L = d.layers[i];
    L.qkv = upload_fused(d, st, {p + "self_attn.q_proj.weight",
                                 p + "self_attn.k_proj.weight",
                                 p + "self_attn.v_proj.weight"}, s.hidden);
    L.o = upload(d, st, p + "self_attn.o_proj.weight", int64_t(s.hidden) * s.q_dim());
    L.gate_up = upload_fused(d, st, {p + "mlp.gate_proj.weight",
                                     p + "mlp.up_proj.weight"}, s.hidden);
    L.down = upload(d, st, p + "mlp.down_proj.weight", int64_t(s.hidden) * d.inter);
    L.attn_kproj = upload(d, st, p + "attention_conv.kernel_projection.weight",
                          int64_t(s.kproj_out()) * s.hidden);
    L.mlp_kproj = upload(d, st, p + "mlp_conv.kernel_projection.weight",
                         int64_t(s.kproj_out()) * s.hidden);
    L.input_ln = upload(d, st, p + "input_layernorm.weight", s.hidden);
    L.post_ln = upload(d, st, p + "post_attention_layernorm.weight", s.hidden);
    L.q_norm = upload(d, st, p + "self_attn.q_norm.weight", s.head_dim);
    L.k_norm = upload(d, st, p + "self_attn.k_norm.weight", s.head_dim);
    L.attn_base = upload(d, st, p + "attention_conv.base_kernel",
                         int64_t(2) * s.conv_k * s.hidden);
    L.mlp_base = upload(d, st, p + "mlp_conv.base_kernel",
                        int64_t(2) * s.conv_k * s.hidden);
  }
  st.close();

  if (d.quantized) {
    const int g = opt.group_size;
    quant_release(d, d.fc, d.fc_q, s.hidden, s.n_taps * s.hidden, g);
    for (int i = 0; i < s.n_layers; ++i) {
      DraftLayer& L = d.layers[i];
      quant_release(d, L.qkv, L.qkv_q, s.qkv_dim(), s.hidden, g);
      quant_release(d, L.o, L.o_q, s.hidden, s.q_dim(), g);
      quant_release(d, L.gate_up, L.gate_up_q, 2 * d.inter, s.hidden, g);
      quant_release(d, L.down, L.down_q, s.hidden, d.inter, g);
      quant_release(d, L.attn_kproj, L.attn_kproj_q, s.kproj_out(), s.hidden, g);
      quant_release(d, L.mlp_kproj, L.mlp_kproj_q, s.kproj_out(), s.hidden, g);
    }
    // The GEMV/MMA path needs its own fp32 scratch.
    gemv_scratch_alloc(d.gemv, s.n_taps * s.hidden, 2 * d.inter, g, 16);
    d.owned.push_back(d.gemv.xf);
    d.owned.push_back(d.gemv.xgsum);
    d.owned.push_back(d.gemv.partial);
    d.bytes += d.gemv.bytes();
  }

  // scratch
  const int R = std::max(opt.ctx_chunk, s.block_size);
  d.max_rows = R;
  const int H = s.hidden;
  d.ctx_h   = static_cast<__nv_bfloat16*>(dmalloc(d, size_t(R) * H * 2));
  d.noise   = static_cast<__nv_bfloat16*>(dmalloc(d, size_t(s.block_size) * H * 2));
  d.h       = static_cast<__nv_bfloat16*>(dmalloc(d, size_t(s.block_size) * H * 2));
  d.h2      = static_cast<__nv_bfloat16*>(dmalloc(d, size_t(s.block_size) * H * 2));
  d.proj    = static_cast<__nv_bfloat16*>(dmalloc(d, size_t(R) * std::max({s.qkv_dim(), 2 * d.inter, s.kproj_out()}) * 2));
  d.qbuf    = static_cast<__nv_bfloat16*>(dmalloc(d, size_t(s.block_size) * s.q_dim() * 2));
  d.kvbuf   = static_cast<__nv_bfloat16*>(dmalloc(d, size_t(R + s.block_size) * 2 * s.kv_dim() * 2));
  d.attn_out= static_cast<__nv_bfloat16*>(dmalloc(d, size_t(s.block_size) * s.q_dim() * 2));
  d.mlp_tmp = static_cast<__nv_bfloat16*>(dmalloc(d, size_t(s.block_size) * d.inter * 2));
  d.dynbuf  = static_cast<__nv_bfloat16*>(dmalloc(d, size_t(s.block_size) * s.kproj_out() * 2));
  d.cbuf    = static_cast<__nv_bfloat16*>(dmalloc(d, size_t(s.block_size) * H * 2));
  d.cos_tab = static_cast<float*>(dmalloc(d, size_t(R + s.block_size) * s.head_dim / 2 * 4));
  d.sin_tab = static_cast<float*>(dmalloc(d, size_t(R + s.block_size) * s.head_dim / 2 * 4));
  d.pos_buf = static_cast<int32_t*>(dmalloc(d, size_t(R + s.block_size) * 4));
  d.ids_buf = static_cast<int32_t*>(dmalloc(d, size_t(s.block_size) * 4));
  d.hp      = static_cast<__nv_bfloat16*>(dmalloc(d, size_t(s.block_size) * s.rank * 2));
  d.unary   = static_cast<float*>(dmalloc(d, size_t(s.block_size) * s.top_k * 4));
  d.cand    = static_cast<int32_t*>(dmalloc(d, size_t(s.block_size) * s.top_k * 4));
  d.path    = static_cast<int32_t*>(dmalloc(d, size_t(s.block_size) * 4));

  d.cache_cap = s.sliding_window > 0 ? 2 * s.sliding_window : R + s.block_size;
  d.k_cache = static_cast<__nv_bfloat16*>(
      dmalloc(d, size_t(s.n_layers) * d.cache_cap * s.kv_dim() * 2));
  d.v_cache = static_cast<__nv_bfloat16*>(
      dmalloc(d, size_t(s.n_layers) * d.cache_cap * s.kv_dim() * 2));

  d.debug = opt.debug;
  if (d.debug) {
    const size_t rows = std::max(R, s.block_size);
    for (int i = 0; i < DraftModel::DBG_N; ++i)
      d.dbg[i] = static_cast<__nv_bfloat16*>(dmalloc(d, rows * H * 2));
  }

  if (!d.cublas) CBD(cublasCreate(&d.cublas));
  CBD(cublasSetStream(d.cublas, d.stream));

  if (opt.verbose) {
    printf("dflash drafter: %d layers, hidden %d, block %d, taps [", s.n_layers,
           s.hidden, s.block_size);
    for (size_t i = 0; i < s.target_layer_ids.size(); ++i)
      printf("%s%d", i ? "," : "", s.target_layer_ids[i]);
    printf("], window %d, %s, %.2f GiB\n", s.sliding_window,
           d.quantized ? "W4A16" : "BF16", double(d.bytes) / (1 << 30));
  }
}

void draft_free(DraftModel& d) {
  for (void* p : d.owned) cudaFree(p);
  d.owned.clear();
  if (d.cublas) { cublasDestroy(d.cublas); d.cublas = nullptr; }
  d.bytes = 0;
}

void draft_reset(DraftModel& d) { d.cache_len = 0; d.cache_pos0 = 0; }


namespace {

// dst[r][0..w) = src[r][off..off+w)
__global__ void k_slice(__nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src,
                        int rows, int w, int dst_stride, int src_stride, int off) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * w) return;
  const int r = int(i / w), c = int(i % w);
  dst[size_t(r) * dst_stride + c] = src[size_t(r) * src_stride + off + c];
}

void slice(DraftModel& d, __nv_bfloat16* dst, const __nv_bfloat16* src, int rows,
           int w, int dst_stride, int src_stride, int off) {
  const size_t n = size_t(rows) * w;
  k_slice<<<(n + 255) / 256, 256, 0, d.stream>>>(dst, src, rows, w, dst_stride,
                                                 src_stride, off);
}

void dyn_conv(DraftModel& d, __nv_bfloat16* out, const __nv_bfloat16* in,
              const __nv_bfloat16* dyn, const __nv_bfloat16* base_kernel, int T,
              int which) {
  const DFlashShape& s = d.sh;
  k_dyn_conv<<<dim3(T, 8), 256, 0, d.stream>>>(
      out, in, dyn, base_kernel + size_t(which) * s.conv_k * s.hidden, T, s.hidden,
      s.conv_group, s.conv_k, s.conv_groups(), which);
}

// The context half of a layer: project K and V from the TARGET's residual
// stream (already fc'd and hidden_norm'd) into the cache. No queries, no
// attention -- context rows are never attended FROM, only TO.
void draft_ctx_layer(DraftModel& d, int li, const __nv_bfloat16* ctx_norm,
                     int n_ctx, int base) {
  const DFlashShape& s = d.sh;
  DraftLayer& L = d.layers[li];
  const int H = s.hidden, HD = s.head_dim, KV = s.kv_dim(), Q = s.q_dim();
  const int QKV = s.qkv_dim();
  __nv_bfloat16* kc = d.k_cache + size_t(li) * d.cache_cap * KV;
  __nv_bfloat16* vc = d.v_cache + size_t(li) * d.cache_cap * KV;

  dlinear(d, d.proj, L.qkv, L.qkv_q, ctx_norm, QKV, H, n_ctx);
  slice(d, kc + size_t(base) * KV, d.proj, n_ctx, KV, KV, QKV, Q);
  slice(d, vc + size_t(base) * KV, d.proj, n_ctx, KV, KV, QKV, Q + KV);
  k_head_norm<<<dim3(n_ctx, s.n_kv_heads), 128, 128 * sizeof(float), d.stream>>>(
      kc + size_t(base) * KV, L.k_norm, s.n_kv_heads, HD, s.rms_eps);
  k_rope_half<<<dim3(n_ctx, s.n_kv_heads), 128, 0, d.stream>>>(
      kc + size_t(base) * KV, d.cos_tab, d.sin_tab, s.n_kv_heads, HD, HD / 2);
}

// The block half: T noise rows, bidirectional within the window.
void draft_block_layer(DraftModel& d, int li, int T, int block_pos0, int nctx) {
  const DFlashShape& s = d.sh;
  DraftLayer& L = d.layers[li];
  const int H = s.hidden, HD = s.head_dim, KV = s.kv_dim(), Q = s.q_dim();
  const int QKV = s.qkv_dim(), half = HD / 2;
  __nv_bfloat16* kc = d.k_cache + size_t(li) * d.cache_cap * KV;
  __nv_bfloat16* vc = d.v_cache + size_t(li) * d.cache_cap * KV;

  rmsnorm_plain(d, d.h2, d.h, L.input_ln, T, H);
  if (d.debug && li == 0) CKD(cudaMemcpyAsync(d.dbg[DraftModel::DBG_L0_LN], d.h2, size_t(T) * H * 2, cudaMemcpyDeviceToDevice, d.stream));
  dlinear(d, d.dynbuf, L.attn_kproj, L.attn_kproj_q, d.h2, s.kproj_out(), H, T);
  dyn_conv(d, d.cbuf, d.h2, d.dynbuf, L.attn_base, T, 0);
  if (d.debug && li == 0) CKD(cudaMemcpyAsync(d.dbg[DraftModel::DBG_L0_CONV], d.cbuf, size_t(T) * H * 2, cudaMemcpyDeviceToDevice, d.stream));
  dlinear(d, d.proj, L.qkv, L.qkv_q, d.cbuf, QKV, H, T);

  slice(d, d.qbuf, d.proj, T, Q, Q, QKV, 0);
  slice(d, kc + size_t(nctx) * KV, d.proj, T, KV, KV, QKV, Q);
  slice(d, vc + size_t(nctx) * KV, d.proj, T, KV, KV, QKV, Q + KV);

  k_head_norm<<<dim3(T, s.n_q_heads), 128, 128 * sizeof(float), d.stream>>>(
      d.qbuf, L.q_norm, s.n_q_heads, HD, s.rms_eps);
  k_head_norm<<<dim3(T, s.n_kv_heads), 128, 128 * sizeof(float), d.stream>>>(
      kc + size_t(nctx) * KV, L.k_norm, s.n_kv_heads, HD, s.rms_eps);
  k_rope_half<<<dim3(T, s.n_q_heads), 128, 0, d.stream>>>(
      d.qbuf, d.cos_tab, d.sin_tab, s.n_q_heads, HD, half);
  k_rope_half<<<dim3(T, s.n_kv_heads), 128, 0, d.stream>>>(
      kc + size_t(nctx) * KV, d.cos_tab, d.sin_tab, s.n_kv_heads, HD, half);

  k_attn_draft<<<s.n_q_heads, 32 * T, 0, d.stream>>>(
      d.attn_out, d.qbuf, kc, vc, T, nctx + T, s.n_q_heads, s.n_kv_heads, HD,
      rsqrtf(float(HD)), s.sliding_window, block_pos0, d.cache_pos0, s.is_causal);

  dlinear(d, d.h2, L.o, L.o_q, d.attn_out, H, Q, T);
  if (d.debug && li == 0) CKD(cudaMemcpyAsync(d.dbg[DraftModel::DBG_L0_ATTN], d.h2, size_t(T) * H * 2, cudaMemcpyDeviceToDevice, d.stream));
  dyn_conv(d, d.cbuf, d.h2, d.dynbuf, L.attn_base, T, 1);
  residual_add(d.h, d.cbuf, T * H, d.stream);

  rmsnorm_plain(d, d.h2, d.h, L.post_ln, T, H);
  if (d.debug && li == 0) CKD(cudaMemcpyAsync(d.dbg[DraftModel::DBG_L0_POST_LN], d.h2, size_t(T) * H * 2, cudaMemcpyDeviceToDevice, d.stream));
  dlinear(d, d.dynbuf, L.mlp_kproj, L.mlp_kproj_q, d.h2, s.kproj_out(), H, T);
  dyn_conv(d, d.cbuf, d.h2, d.dynbuf, L.mlp_base, T, 0);
  dlinear(d, d.proj, L.gate_up, L.gate_up_q, d.cbuf, 2 * d.inter, H, T);
  swiglu(d.mlp_tmp, d.proj, T, d.inter, d.stream);
  dlinear(d, d.h2, L.down, L.down_q, d.mlp_tmp, H, d.inter, T);
  dyn_conv(d, d.cbuf, d.h2, d.dynbuf, L.mlp_base, T, 1);
  residual_add(d.h, d.cbuf, T * H, d.stream);
  if (d.debug && li == 0) CKD(cudaMemcpyAsync(d.dbg[DraftModel::DBG_L0_OUT], d.h, size_t(T) * H * 2, cudaMemcpyDeviceToDevice, d.stream));
}

void rope_for(DraftModel& d, int pos0, int n) {
  std::vector<int32_t> pos(n);
  for (int i = 0; i < n; ++i) pos[i] = pos0 + i;
  CKD(cudaMemcpyAsync(d.pos_buf, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice,
                      d.stream));
  const int half = d.sh.head_dim / 2;
  const int total = n * half;
  k_rope_tables<<<(total + 255) / 256, 256, 0, d.stream>>>(
      d.cos_tab, d.sin_tab, d.pos_buf, n, half, d.sh.rope_theta);
}

}  // namespace

void draft_push(DraftModel& d, const __nv_bfloat16* target_hidden, int n_ctx,
                int ctx_pos0) {
  const DFlashShape& s = d.sh;
  const int H = s.hidden, T = s.block_size;
  if (n_ctx <= 0) return;
  if (n_ctx > d.max_rows) {
    fprintf(stderr, "dflash: %d context rows exceeds ctx_chunk %d\n", n_ctx, d.max_rows);
    abort();
  }
  if (d.cache_len == 0 && d.cache_pos0 != ctx_pos0) d.cache_pos0 = ctx_pos0;
  if (d.cache_pos0 + d.cache_len != ctx_pos0) {
    fprintf(stderr, "dflash: cache ends at %d but context starts at %d\n",
            d.cache_pos0 + d.cache_len, ctx_pos0);
    abort();
  }
  if (d.cache_len + n_ctx + T > d.cache_cap) {
    // Keep the last window-1 slots and drop the rest. At 2*window capacity this
    // memmove happens once per ~window committed tokens; anything older than
    // the window is masked out anyway, so nothing is lost.
    const int keep = std::min(d.cache_len, std::max(0, s.sliding_window - 1));
    const int drop = d.cache_len - keep;
    for (int li = 0; li < s.n_layers; ++li) {
      __nv_bfloat16* kc = d.k_cache + size_t(li) * d.cache_cap * s.kv_dim();
      __nv_bfloat16* vc = d.v_cache + size_t(li) * d.cache_cap * s.kv_dim();
      CKD(cudaMemcpyAsync(kc, kc + size_t(drop) * s.kv_dim(),
                          size_t(keep) * s.kv_dim() * 2, cudaMemcpyDeviceToDevice, d.stream));
      CKD(cudaMemcpyAsync(vc, vc + size_t(drop) * s.kv_dim(),
                          size_t(keep) * s.kv_dim() * 2, cudaMemcpyDeviceToDevice, d.stream));
    }
    d.cache_pos0 += drop;
    d.cache_len = keep;
  }

  // The reference does hidden_norm(fc(target_hidden)) once; every layer reads
  // the same context tensor.
  dlinear(d, d.ctx_h, d.fc, d.fc_q, target_hidden, H, s.n_taps * H, n_ctx);
  rmsnorm_plain(d, d.ctx_h, d.ctx_h, d.hidden_norm, n_ctx, H);
  if (d.debug) CKD(cudaMemcpyAsync(d.dbg[DraftModel::DBG_CTX_NORM], d.ctx_h,
                                   size_t(n_ctx) * H * 2, cudaMemcpyDeviceToDevice, d.stream));
  rope_for(d, ctx_pos0, n_ctx);
  for (int li = 0; li < s.n_layers; ++li)
    draft_ctx_layer(d, li, d.ctx_h, n_ctx, d.cache_len);
  d.cache_len += n_ctx;
}

const __nv_bfloat16* draft_block(DraftModel& d, int block_pos0) {
  const DFlashShape& s = d.sh;
  const int T = s.block_size, H = s.hidden;
  if (32 * T > 1024) { fprintf(stderr, "dflash: block %d too large\n", T); abort(); }
  if (d.cache_pos0 + d.cache_len != block_pos0) {
    fprintf(stderr, "dflash: cache ends at %d but block starts at %d\n",
            d.cache_pos0 + d.cache_len, block_pos0);
    abort();
  }
  rope_for(d, block_pos0, T);
  CKD(cudaMemcpyAsync(d.h, d.noise, size_t(T) * H * 2, cudaMemcpyDeviceToDevice, d.stream));
  for (int li = 0; li < s.n_layers; ++li)
    draft_block_layer(d, li, T, block_pos0, d.cache_len);
  rmsnorm_plain(d, d.h2, d.h, d.final_norm, T, H);
  return d.h2;
}

const __nv_bfloat16* draft_forward(DraftModel& d, const __nv_bfloat16* target_hidden,
                                   int n_ctx, int ctx_pos0, int block_pos0,
                                   bool use_cache) {
  if (!use_cache) draft_reset(d);
  if (!use_cache) d.cache_pos0 = ctx_pos0;
  draft_push(d, target_hidden, n_ctx, ctx_pos0);
  return draft_block(d, block_pos0);
}

void draft_select(DraftModel& d, const __nv_bfloat16* draft_hidden,
                  const __nv_bfloat16* logits, int n_pos, int32_t anchor_id) {
  const DFlashShape& s = d.sh;
  // hidden_projection stays bf16: it is 2.6 MB, and quantising the selector's
  // own projection would degrade draft quality for no bandwidth worth having.
  const float one = 1.f, zero = 0.f;
  CBD(cublasGemmEx(d.cublas, CUBLAS_OP_T, CUBLAS_OP_N, s.rank, n_pos, s.hidden, &one,
                   d.hproj, CUDA_R_16BF, s.hidden, draft_hidden, CUDA_R_16BF, s.hidden,
                   &zero, d.hp, CUDA_R_16BF, s.rank, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  k_topk<<<n_pos, 256, 0, d.stream>>>(d.unary, d.cand, logits, s.vocab, s.top_k);
  const size_t shm = (size_t(s.rank) + 256) * sizeof(float);
  k_selector<<<1, 256, shm, d.stream>>>(d.path, d.hp, d.unary, d.cand, d.pred_cb,
                                        d.succ_cb, n_pos, s.top_k, s.rank, anchor_id);
}

}  // namespace qwen
