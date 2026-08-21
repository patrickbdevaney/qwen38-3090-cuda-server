#include "vit.h"
#include "../loader/safetensors.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <fstream>
#include "../../third_party/json.hpp"

namespace qwen {
namespace {

#define CKV(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); abort(); } } while(0)
#define CBV(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS %s:%d status %d\n",__FILE__,__LINE__,int(s_)); abort(); } } while(0)

// LayerNorm WITH bias, fp32 accumulation. Not RMSNorm: the vision tower is a
// stock ViT and subtracts the mean.
__global__ void k_layernorm(__nv_bfloat16* __restrict__ out,
                            const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ w,
                            const __nv_bfloat16* __restrict__ b,
                            int dim, float eps) {
  extern __shared__ float red[];
  const __nv_bfloat16* row = x + size_t(blockIdx.x) * dim;
  __nv_bfloat16* dst = out + size_t(blockIdx.x) * dim;
  float s = 0.f, ss = 0.f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = __bfloat162float(row[i]);
    s += v; ss += v * v;
  }
  red[threadIdx.x] = s; red[threadIdx.x + blockDim.x] = ss;
  __syncthreads();
  for (int k = blockDim.x >> 1; k > 0; k >>= 1) {
    if (threadIdx.x < k) {
      red[threadIdx.x] += red[threadIdx.x + k];
      red[threadIdx.x + blockDim.x] += red[threadIdx.x + blockDim.x + k];
    }
    __syncthreads();
  }
  const float mean = red[0] / float(dim);
  const float var = red[blockDim.x] / float(dim) - mean * mean;
  const float inv = rsqrtf(var + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = (__bfloat162float(row[i]) - mean) * inv;
    dst[i] = __float2bfloat16(v * __bfloat162float(w[i]) + __bfloat162float(b[i]));
  }
}

// out[i] += bias[i % dim]
__global__ void k_add_bias(__nv_bfloat16* __restrict__ x,
                           const __nv_bfloat16* __restrict__ b, int n, int dim) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(n) * dim) return;
  x[i] = __hadd(x[i], b[i % dim]);
}

// gelu_pytorch_tanh -- the block MLP's activation.
__global__ void k_gelu_tanh(__nv_bfloat16* __restrict__ x, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = __bfloat162float(x[i]);
  const float c = 0.7978845608028654f;   // sqrt(2/pi)
  x[i] = __float2bfloat16(0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v))));
}

// Exact erf GELU -- the MERGER's activation. nn.GELU() is not
// gelu_pytorch_tanh, and using one for the other is a quiet ~1e-3 error.
__global__ void k_gelu_erf(__nv_bfloat16* __restrict__ x, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = __bfloat162float(x[i]);
  x[i] = __float2bfloat16(0.5f * v * (1.0f + erff(v * 0.7071067811865476f)));
}

__global__ void k_residual(__nv_bfloat16* __restrict__ a,
                           const __nv_bfloat16* __restrict__ b, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) a[i] = __hadd(a[i], b[i]);
}

// Patch positions in SPATIAL-MERGE-BLOCK order, and the bilinear resample of the
// learned pos_embed onto this grid, fused into one pass.
//
//   within -> (block_row, block_col, in_row, in_col) -> (row, col)
//   src    = row * (side - 1) / max(h - 1, 1)    [align_corners = TRUE]
//
// The model sets interpolation_align_corners = True, which is the endpoint-
// matching form, NOT the half-pixel-centre form. They differ by up to half a
// grid cell, which is a small, entirely plausible-looking wrong answer.
__global__ void k_pos_embed_add(__nv_bfloat16* __restrict__ x,
                                const __nv_bfloat16* __restrict__ table,
                                int32_t* __restrict__ pos_ids,
                                int n, int hidden, int gh, int gw, int merge,
                                int side) {
  const int p = blockIdx.x;
  if (p >= n) return;
  const int hw = gh * gw;
  const int within = p % hw;
  const int blocks_w = gw / merge;
  const int in_col = within % merge;
  const int in_row = (within / merge) % merge;
  const int block_col = (within / (merge * merge)) % blocks_w;
  const int block_row = within / (merge * merge * blocks_w);
  const int row = block_row * merge + in_row;
  const int col = block_col * merge + in_col;
  if (threadIdx.x == 0) { pos_ids[2 * p] = row; pos_ids[2 * p + 1] = col; }

  const float sy = float(row) * float(side - 1) / float(max(gh - 1, 1));
  const float sx = float(col) * float(side - 1) / float(max(gw - 1, 1));
  const float fy = floorf(sy), fx = floorf(sx);
  const float wy = sy - fy, wx = sx - fx;
  const int y0 = min(max(int(fy), 0), side - 1), y1 = min(max(int(fy) + 1, 0), side - 1);
  const int x0 = min(max(int(fx), 0), side - 1), x1 = min(max(int(fx) + 1, 0), side - 1);
  // weights come from the distance, not from the clamped taps, so they still
  // sum to 1 at the edges
  const float w00 = (1 - wy) * (1 - wx), w01 = (1 - wy) * wx;
  const float w10 = wy * (1 - wx), w11 = wy * wx;

  __nv_bfloat16* dst = x + size_t(p) * hidden;
  const __nv_bfloat16* r00 = table + size_t(y0 * side + x0) * hidden;
  const __nv_bfloat16* r01 = table + size_t(y0 * side + x1) * hidden;
  const __nv_bfloat16* r10 = table + size_t(y1 * side + x0) * hidden;
  const __nv_bfloat16* r11 = table + size_t(y1 * side + x1) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float pe = w00 * __bfloat162float(r00[i]) + w01 * __bfloat162float(r01[i]) +
                     w10 * __bfloat162float(r10[i]) + w11 * __bfloat162float(r11[i]);
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + pe);
  }
}

// 2D rope tables. inv_freq has rope_dim/2 entries; (row, col) each contribute
// that many, giving rope_dim, which is then duplicated to head_dim by the
// rotate_half convention.
__global__ void k_vis_rope_tables(float* __restrict__ cs, float* __restrict__ sn,
                                  const int32_t* __restrict__ pos, int n,
                                  int rope_dim, float theta) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * rope_dim) return;
  const int p = i / rope_dim, d = i % rope_dim;
  const int half = rope_dim / 2;
  const int axis = d / half;          // 0 = row, 1 = col
  const int k = d % half;
  const double inv = 1.0 / pow(double(theta), (2.0 * k) / double(rope_dim));
  const float ang = float(pos[2 * p + axis]) * float(inv);
  cs[i] = cosf(ang);
  sn[i] = sinf(ang);
}

// Apply rope to q and k inside the fused qkv buffer, which is laid out
// [n][3][heads][head_dim].
__global__ void k_vis_rope(__nv_bfloat16* __restrict__ qkv,
                           const float* __restrict__ cs, const float* __restrict__ sn,
                           int heads, int hd, int rope_dim) {
  const int p = blockIdx.x, h = blockIdx.y, which = blockIdx.z;  // which: 0=q, 1=k
  const size_t base = (size_t(p) * 3 + which) * heads * hd + size_t(h) * hd;
  const float* c = cs + size_t(p) * rope_dim;
  const float* s = sn + size_t(p) * rope_dim;
  const int half = hd / 2;
  for (int i = threadIdx.x; i < half; i += blockDim.x) {
    // emb = cat(rope, rope) so the second half reuses the same angle
    const float ci = c[i % rope_dim], si = s[i % rope_dim];
    const float a = __bfloat162float(qkv[base + i]);
    const float b = __bfloat162float(qkv[base + i + half]);
    qkv[base + i]        = __float2bfloat16(a * ci - b * si);
    qkv[base + i + half] = __float2bfloat16(b * ci + a * si);
  }
}

// Row-wise softmax over `nk` keys, fp32 in, bf16 out.
__global__ void k_softmax_rows(__nv_bfloat16* __restrict__ p,
                               const float* __restrict__ s, int nk, float scale) {
  extern __shared__ float red[];
  const float* row = s + size_t(blockIdx.x) * nk;
  __nv_bfloat16* dst = p + size_t(blockIdx.x) * nk;
  float m = -INFINITY;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) m = fmaxf(m, row[i] * scale);
  red[threadIdx.x] = m; __syncthreads();
  for (int k = blockDim.x >> 1; k > 0; k >>= 1) {
    if (threadIdx.x < k) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + k]);
    __syncthreads();
  }
  m = red[0]; __syncthreads();
  float sum = 0.f;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) sum += __expf(row[i] * scale - m);
  red[threadIdx.x] = sum; __syncthreads();
  for (int k = blockDim.x >> 1; k > 0; k >>= 1) {
    if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k];
    __syncthreads();
  }
  const float inv = 1.f / red[0];
  for (int i = threadIdx.x; i < nk; i += blockDim.x)
    dst[i] = __float2bfloat16(__expf(row[i] * scale - m) * inv);
}

void* vmalloc(VisionTower& v, size_t bytes) {
  void* p = nullptr;
  CKV(cudaMalloc(&p, bytes));
  v.owned.push_back(p);
  v.bytes += bytes;
  return p;
}

__nv_bfloat16* vup(VisionTower& v, const SafeTensors& st, const std::string& n,
                   int64_t expect) {
  const TensorView& t = st.get(n);
  if (expect > 0 && t.numel() != expect) {
    fprintf(stderr, "vision: %s has %lld elems, expected %lld\n", n.c_str(),
            (long long)t.numel(), (long long)expect);
    abort();
  }
  auto* p = static_cast<__nv_bfloat16*>(vmalloc(v, t.nbytes));
  CKV(cudaMemcpy(p, t.data, t.nbytes, cudaMemcpyHostToDevice));
  return p;
}

}  // namespace

// y[M,out] = x[M,in] @ W^T + b, W row-major [out, in].
static void vlinear(VisionTower& v, __nv_bfloat16* y, const __nv_bfloat16* W,
                    const __nv_bfloat16* b, const __nv_bfloat16* x,
                    int out_f, int in_f, int M) {
  const float one = 1.f, zero = 0.f;
  CBV(cublasGemmEx(v.cublas, CUBLAS_OP_T, CUBLAS_OP_N, out_f, M, in_f, &one,
                   W, CUDA_R_16BF, in_f, x, CUDA_R_16BF, in_f, &zero,
                   y, CUDA_R_16BF, out_f, CUBLAS_COMPUTE_32F,
                   CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  if (b) {
    const size_t n = size_t(M) * out_f;
    k_add_bias<<<(n + 255) / 256, 256, 0, v.stream>>>(y, b, M, out_f);
  }
}

static void vnorm(VisionTower& v, __nv_bfloat16* out, const __nv_bfloat16* x,
                  const __nv_bfloat16* w, const __nv_bfloat16* b, int rows, int dim) {
  const int th = 256;
  k_layernorm<<<rows, th, 2 * th * sizeof(float), v.stream>>>(out, x, w, b, dim, v.sh.eps);
}

void vision_load(VisionTower& v, const std::string& dir, const VisionLoadOptions& opt) {
  using json = nlohmann::json;
  json cfg;
  { std::ifstream f(dir + "/config.json");
    if (!f) { fprintf(stderr, "vision: no config.json in %s\n", dir.c_str()); abort(); }
    f >> cfg; }
  const json& vc = cfg.at("vision_config");
  VisionShape& s = v.sh;
  s.depth = vc.at("depth");
  s.hidden = vc.at("hidden_size");
  s.intermediate = vc.at("intermediate_size");
  s.num_heads = vc.at("num_heads");
  s.head_dim = s.hidden / s.num_heads;
  s.in_channels = vc.at("in_channels");
  s.patch_size = vc.at("patch_size");
  s.temporal_patch = vc.at("temporal_patch_size");
  s.spatial_merge = vc.at("spatial_merge_size");
  s.num_pos_embed = vc.at("num_position_embeddings");
  s.grid_per_side = int(std::lround(std::sqrt(double(s.num_pos_embed))));
  s.out_hidden = vc.at("out_hidden_size");
  if (s.grid_per_side * s.grid_per_side != s.num_pos_embed) {
    fprintf(stderr, "vision: num_position_embeddings %d is not a square\n", s.num_pos_embed);
    abort();
  }

  SafeTensors st;
  st.open_dir(dir);
  const std::string P = "model.visual.";
  v.patch_w = vup(v, st, P + "patch_embed.proj.weight", int64_t(s.hidden) * s.patch_dim());
  v.patch_b = vup(v, st, P + "patch_embed.proj.bias", s.hidden);
  v.pos_embed = vup(v, st, P + "pos_embed.weight", int64_t(s.num_pos_embed) * s.hidden);
  v.merger_norm_w = vup(v, st, P + "merger.norm.weight", s.hidden);
  v.merger_norm_b = vup(v, st, P + "merger.norm.bias", s.hidden);
  v.merger_fc1_w = vup(v, st, P + "merger.linear_fc1.weight",
                       int64_t(s.merged_dim()) * s.merged_dim());
  v.merger_fc1_b = vup(v, st, P + "merger.linear_fc1.bias", s.merged_dim());
  v.merger_fc2_w = vup(v, st, P + "merger.linear_fc2.weight",
                       int64_t(s.out_hidden) * s.merged_dim());
  v.merger_fc2_b = vup(v, st, P + "merger.linear_fc2.bias", s.out_hidden);

  v.blocks.resize(s.depth);
  for (int i = 0; i < s.depth; ++i) {
    const std::string b = P + "blocks." + std::to_string(i) + ".";
    VisionBlock& B = v.blocks[i];
    B.norm1_w = vup(v, st, b + "norm1.weight", s.hidden);
    B.norm1_b = vup(v, st, b + "norm1.bias", s.hidden);
    B.norm2_w = vup(v, st, b + "norm2.weight", s.hidden);
    B.norm2_b = vup(v, st, b + "norm2.bias", s.hidden);
    B.qkv_w = vup(v, st, b + "attn.qkv.weight", int64_t(3) * s.hidden * s.hidden);
    B.qkv_b = vup(v, st, b + "attn.qkv.bias", int64_t(3) * s.hidden);
    B.proj_w = vup(v, st, b + "attn.proj.weight", int64_t(s.hidden) * s.hidden);
    B.proj_b = vup(v, st, b + "attn.proj.bias", s.hidden);
    B.fc1_w = vup(v, st, b + "mlp.linear_fc1.weight", int64_t(s.intermediate) * s.hidden);
    B.fc1_b = vup(v, st, b + "mlp.linear_fc1.bias", s.intermediate);
    B.fc2_w = vup(v, st, b + "mlp.linear_fc2.weight", int64_t(s.hidden) * s.intermediate);
    B.fc2_b = vup(v, st, b + "mlp.linear_fc2.bias", s.hidden);
  }
  st.close();

  const int N = opt.max_patches;
  v.max_patches = N;
  v.qtile = std::min(512, N);
  v.x      = static_cast<__nv_bfloat16*>(vmalloc(v, size_t(N) * s.hidden * 2));
  v.xn     = static_cast<__nv_bfloat16*>(vmalloc(v, size_t(N) * s.hidden * 2));
  v.qkv    = static_cast<__nv_bfloat16*>(vmalloc(v, size_t(N) * 3 * s.hidden * 2));
  v.attn   = static_cast<__nv_bfloat16*>(vmalloc(v, size_t(N) * s.hidden * 2));
  v.mlp    = static_cast<__nv_bfloat16*>(vmalloc(v, size_t(N) * s.intermediate * 2));
  v.probs  = static_cast<__nv_bfloat16*>(vmalloc(v, size_t(v.qtile) * N * 2));
  v.scores = static_cast<float*>(vmalloc(v, size_t(v.qtile) * N * 4));
  const int nmax = N / (s.spatial_merge * s.spatial_merge) + 1;
  v.merged = static_cast<__nv_bfloat16*>(vmalloc(v, size_t(nmax) * s.merged_dim() * 2));
  // Separate from v.attn: the merged output is [N/4, 5120] = 327680 elements at
  // N=256, while v.attn is [N, 1152] = 294912. Reusing it overruns.
  v.out = static_cast<__nv_bfloat16*>(vmalloc(v, size_t(nmax) * s.out_hidden * 2));
  v.cos_tab = static_cast<float*>(vmalloc(v, size_t(N) * s.rope_dim() * 4));
  v.sin_tab = static_cast<float*>(vmalloc(v, size_t(N) * s.rope_dim() * 4));
  v.pos_ids = static_cast<int32_t*>(vmalloc(v, size_t(N) * 2 * 4));

  v.debug = opt.debug;
  if (v.debug)
    v.dbg_post_pos = static_cast<__nv_bfloat16*>(vmalloc(v, size_t(N) * s.hidden * 2));

  if (!v.cublas) CBV(cublasCreate(&v.cublas));
  CBV(cublasSetStream(v.cublas, v.stream));

  if (opt.verbose)
    printf("vision tower: %d blocks, hidden %d, %d heads x %d, merge %d, "
           "max %d patches (%d tokens), %.3f GiB\n",
           s.depth, s.hidden, s.num_heads, s.head_dim, s.spatial_merge, N,
           N / (s.spatial_merge * s.spatial_merge), double(v.bytes) / (1 << 30));
}

void vision_free(VisionTower& v) {
  for (void* p : v.owned) cudaFree(p);
  v.owned.clear();
  if (v.cublas) { cublasDestroy(v.cublas); v.cublas = nullptr; }
  v.bytes = 0;
}

namespace {

// Full bidirectional attention over all patches of one image, one head at a
// time, with the queries tiled so the score buffer stays small.
void vision_attention(VisionTower& v, int N) {
  const VisionShape& s = v.sh;
  const int HD = s.head_dim, H = s.num_heads, hid = s.hidden;
  const int qkv_stride = 3 * hid;
  const float scale = 1.0f / sqrtf(float(HD));
  const float one = 1.f, zero = 0.f;

  for (int h = 0; h < H; ++h) {
    const __nv_bfloat16* Q = v.qkv + size_t(0) * hid + size_t(h) * HD;
    const __nv_bfloat16* K = v.qkv + size_t(1) * hid + size_t(h) * HD;
    const __nv_bfloat16* V = v.qkv + size_t(2) * hid + size_t(h) * HD;
    for (int q0 = 0; q0 < N; q0 += v.qtile) {
      const int nq = std::min(v.qtile, N - q0);
      // scores[q][k] = sum_d Q[q][d] K[k][d], written row-major [nq][N]
      CBV(cublasGemmEx(v.cublas, CUBLAS_OP_T, CUBLAS_OP_N, N, nq, HD, &one,
                       K, CUDA_R_16BF, qkv_stride,
                       Q + size_t(q0) * qkv_stride, CUDA_R_16BF, qkv_stride,
                       &zero, v.scores, CUDA_R_32F, N,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      const int th = 256;
      k_softmax_rows<<<nq, th, th * sizeof(float), v.stream>>>(v.probs, v.scores, N, scale);
      // out[q][d] = sum_k P[q][k] V[k][d], written into the packed [N][hidden]
      // buffer at this head's column offset.
      CBV(cublasGemmEx(v.cublas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nq, N, &one,
                       V, CUDA_R_16BF, qkv_stride,
                       v.probs, CUDA_R_16BF, N, &zero,
                       v.attn + size_t(q0) * hid + size_t(h) * HD, CUDA_R_16BF, hid,
                       CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
  }
}

}  // namespace

const __nv_bfloat16* vision_forward(VisionTower& v, const __nv_bfloat16* pixel_values,
                                    int gt, int gh, int gw) {
  const VisionShape& s = v.sh;
  const int N = gt * gh * gw;
  if (N > v.max_patches) {
    fprintf(stderr, "vision: %d patches exceeds max_patches %d\n", N, v.max_patches);
    abort();
  }
  if (gh % s.spatial_merge || gw % s.spatial_merge) {
    fprintf(stderr, "vision: grid %dx%d not a multiple of merge %d\n", gh, gw, s.spatial_merge);
    abort();
  }
  const int hid = s.hidden;

  // patch embed: the Conv3d kernel equals its stride, so it is a linear map on
  // the flattened patch.
  vlinear(v, v.x, v.patch_w, v.patch_b, pixel_values, hid, s.patch_dim(), N);

  // learned pos_embed, bilinearly resampled onto this grid; also emits the
  // (row, col) ids the rotary embedding needs.
  k_pos_embed_add<<<N, 256, 0, v.stream>>>(v.x, v.pos_embed, v.pos_ids, N, hid,
                                           gh, gw, s.spatial_merge, s.grid_per_side);
  if (v.debug)
    CKV(cudaMemcpyAsync(v.dbg_post_pos, v.x, size_t(N) * hid * 2,
                        cudaMemcpyDeviceToDevice, v.stream));
  {
    const int rd = s.rope_dim();
    const int n = N * rd;
    k_vis_rope_tables<<<(n + 255) / 256, 256, 0, v.stream>>>(
        v.cos_tab, v.sin_tab, v.pos_ids, N, rd, s.rope_theta);
  }

  for (int i = 0; i < s.depth; ++i) {
    VisionBlock& B = v.blocks[i];
    vnorm(v, v.xn, v.x, B.norm1_w, B.norm1_b, N, hid);
    vlinear(v, v.qkv, B.qkv_w, B.qkv_b, v.xn, 3 * hid, hid, N);
    k_vis_rope<<<dim3(N, s.num_heads, 2), 64, 0, v.stream>>>(
        v.qkv, v.cos_tab, v.sin_tab, s.num_heads, s.head_dim, s.rope_dim());
    vision_attention(v, N);
    vlinear(v, v.xn, B.proj_w, B.proj_b, v.attn, hid, hid, N);
    k_residual<<<(size_t(N) * hid + 255) / 256, 256, 0, v.stream>>>(
        v.x, v.xn, size_t(N) * hid);

    vnorm(v, v.xn, v.x, B.norm2_w, B.norm2_b, N, hid);
    vlinear(v, v.mlp, B.fc1_w, B.fc1_b, v.xn, s.intermediate, hid, N);
    k_gelu_tanh<<<(size_t(N) * s.intermediate + 255) / 256, 256, 0, v.stream>>>(
        v.mlp, size_t(N) * s.intermediate);
    vlinear(v, v.xn, B.fc2_w, B.fc2_b, v.mlp, hid, s.intermediate, N);
    k_residual<<<(size_t(N) * hid + 255) / 256, 256, 0, v.stream>>>(
        v.x, v.xn, size_t(N) * hid);
  }

  // merger: LayerNorm over hidden, then the 2x2 merge is a pure reshape because
  // patches are already in spatial-merge-block order.
  vnorm(v, v.xn, v.x, v.merger_norm_w, v.merger_norm_b, N, hid);
  const int nout = N / (s.spatial_merge * s.spatial_merge);
  vlinear(v, v.merged, v.merger_fc1_w, v.merger_fc1_b, v.xn,
          s.merged_dim(), s.merged_dim(), nout);
  k_gelu_erf<<<(size_t(nout) * s.merged_dim() + 255) / 256, 256, 0, v.stream>>>(
      v.merged, size_t(nout) * s.merged_dim());
  vlinear(v, v.out, v.merger_fc2_w, v.merger_fc2_b, v.merged,
          s.out_hidden, s.merged_dim(), nout);
  return v.out;
}

}  // namespace qwen
