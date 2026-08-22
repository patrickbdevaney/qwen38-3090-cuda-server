#include "gdn.cuh"
#include <cstdio>
#include <cuda_runtime.h>

namespace qwen {
namespace {

// expf, not __expf. The delta rule computes (v_t - g*S^T k), a residual, so
// cancellation amplifies any input rounding; the ~2 ulp of the fast intrinsics
// is not free here. Measured: it moves the recurrent state.
__device__ __forceinline__ float silu(float x) { return x / (1.f + expf(-x)); }

// PyTorch's F.softplus: log1p(exp(x)), switching to the identity above
// threshold=20 to avoid overflow. Matching the threshold matters because g is
// exponentiated afterwards.
__device__ __forceinline__ float softplus(float x) {
  return x > 20.f ? x : log1pf(expf(x));
}

// ---------------------------------------------------------------- conv + silu
// Depthwise causal conv, one thread per channel. conv_state holds the previous
// conv_k-1 inputs. F.conv1d with padding=K-1 then [:, :, :T] is
//   out[c][t] = sum_j w[c][j] * in[c][t + j - (K-1)]
// with zeros before the start, which the state supplies during decode.
__global__ void k_conv(__nv_bfloat16* __restrict__ out, float* __restrict__ state,
                       const __nv_bfloat16* __restrict__ in,
                       const __nv_bfloat16* __restrict__ w,
                       int conv_dim, int K, int T, bool use_state, int in_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;

  float hist[8];                       // K <= 8
  for (int j = 0; j < K - 1; ++j)
    hist[j] = use_state ? state[size_t(c) * (K - 1) + j] : 0.f;

  float wc[8];
  for (int j = 0; j < K; ++j) wc[j] = __bfloat162float(w[size_t(c) * K + j]);

  for (int t = 0; t < T; ++t) {
    const float x = __bfloat162float(in[size_t(t) * in_stride + c]);
    // Accumulate oldest-to-newest, the order F.conv1d uses. fp32 addition is not
    // associative, and a different order rounds to a different bf16 output,
    // which the delta rule's (v_t - g*S^T k) cancellation then amplifies.
    float acc = 0.f;
    for (int j = 0; j < K - 1; ++j) acc += wc[j] * hist[j];
    acc += wc[K - 1] * x;
    out[size_t(t) * conv_dim + c] = __float2bfloat16(silu(acc));
    for (int j = 0; j < K - 2; ++j) hist[j] = hist[j + 1];
    hist[K - 2] = x;
  }
  for (int j = 0; j < K - 1; ++j) state[size_t(c) * (K - 1) + j] = hist[j];
}

// ---------------------------------------------------------------- gates
__global__ void k_gates(float* __restrict__ g, float* __restrict__ beta,
                        const __nv_bfloat16* __restrict__ a,
                        const __nv_bfloat16* __restrict__ b,
                        const __nv_bfloat16* __restrict__ A_log,
                        const __nv_bfloat16* __restrict__ dt_bias,
                        int T, int H) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * H) return;
  const int h = i % H;
  const float av = __bfloat162float(a[i]);
  const float bv = __bfloat162float(b[i]);
  const float A  = __bfloat162float(A_log[h]);
  const float dt = __bfloat162float(dt_bias[h]);
  // g is computed and kept in fp32: the reference is explicit that A overflows
  // to -inf in fp16 otherwise.
  g[i] = -expf(A) * softplus(av + dt);
  // beta is NOT. The reference does `b.sigmoid()` on a bf16 tensor, so its beta
  // carries only 8 mantissa bits before being widened to fp32 inside the
  // recurrence. Computing it in fp32 here is strictly more accurate and would
  // put us 3.7e-3 away from the reference -- enough to flip an argmax over 64
  // layers. Round through bf16 to match.
  beta[i] = __bfloat162float(__float2bfloat16(1.f / (1.f + expf(-bv))));
}

// ---------------------------------------------------------------- scan
// One block per value head. S lives in shared memory as [head_k][head_v] fp32,
// 64 KB at 128x128, which is why this is one block per SM. Threads are laid out
// as (v = tid % head_v, kgroup = tid / head_v); with 256 threads each thread
// owns head_k/2 = 64 k-rows for its v column.
//
// Shared-memory banking: a thread reads S[k][v] at offset k*head_v + v, so the
// bank is v % 32. Lanes of a warp have consecutive v, hence 32 distinct banks
// and no conflict, on both the read and the writeback.
template <int HEAD_K, int V_PER_BLOCK, int THREADS, int TB>
__global__ __launch_bounds__(THREADS) void k_scan(
    __nv_bfloat16* __restrict__ out, float* __restrict__ state,
    const __nv_bfloat16* __restrict__ qkv,
    const float* __restrict__ gs, const float* __restrict__ betas,
    int T, int conv_dim, int key_dim, int val_dim, int num_v_heads,
    int hk_div, int hk_mod,
    int head_v, int v_blocks) {
  extern __shared__ float sm[];
  // Stride padded by 1: without it a thread reads S[k][v] at k*VPB + v, so the
  // bank is (j*VPB + v) % 32 and threads in different k-groups but the same v
  // collide. VPB+1 makes the bank vary with k.
  constexpr int SSTRIDE = V_PER_BLOCK + 1;
  float* S   = sm;                                  // [HEAD_K][SSTRIDE]
  float* red = sm + HEAD_K * SSTRIDE;               // A/B partials + preamble scratch
  __shared__ float sb_q[TB * HEAD_K], sb_k[TB * HEAD_K];
  __shared__ float sb_v[TB * V_PER_BLOCK], sb_kq[TB];

  const int h    = blockIdx.x / v_blocks;           // value head
  const int vb   = blockIdx.x % v_blocks;
  const int vbase = vb * V_PER_BLOCK;
  const int hk   = (h / hk_div) % hk_mod;           // the q/k head it shares

  constexpr int KG  = THREADS / V_PER_BLOCK;        // k-groups
  constexpr int KPT = HEAD_K / KG;                  // k-rows per thread
  const int v  = threadIdx.x % V_PER_BLOCK;
  const int kg = threadIdx.x / V_PER_BLOCK;
  const int k0 = kg * KPT;

  const size_t sbase = (size_t(h) * HEAD_K) * head_v + vbase;
  for (int i = threadIdx.x; i < HEAD_K * V_PER_BLOCK; i += THREADS) {
    const int kk = i / V_PER_BLOCK, vv = i % V_PER_BLOCK;
    S[kk * SSTRIDE + vv] = state[sbase + size_t(kk) * head_v + vv];
  }
  __syncthreads();

  const __nv_bfloat16* qbase = qkv;
  const __nv_bfloat16* kbase = qkv + key_dim;
  const __nv_bfloat16* vbase_p = qkv + 2 * key_dim;

  // Timesteps are staged TB at a time. One at a time, every step paid a full
  // dependent global round trip (~400 cycles) on the critical path
  // stage -> sync -> normalize -> sync -> scan, and at 4096 prefill steps that
  // latency, not the arithmetic, was the cost. Batching also lets the l2norm for
  // all TB steps be done once, which halves the barriers in the inner loop.
  for (int t0 = 0; t0 < T; t0 += TB) {
    const int tb = min(TB, T - t0);

    for (int i = threadIdx.x; i < tb * HEAD_K; i += THREADS) {
      const int tt = i / HEAD_K, idx = i % HEAD_K;
      sb_q[tt * HEAD_K + idx] =
          __bfloat162float(qbase[size_t(t0 + tt) * conv_dim + hk * HEAD_K + idx]);
      sb_k[tt * HEAD_K + idx] =
          __bfloat162float(kbase[size_t(t0 + tt) * conv_dim + hk * HEAD_K + idx]);
    }
    for (int i = threadIdx.x; i < tb * V_PER_BLOCK; i += THREADS) {
      const int tt = i / V_PER_BLOCK, idx = i % V_PER_BLOCK;
      sb_v[tt * V_PER_BLOCK + idx] = __bfloat162float(
          vbase_p[size_t(t0 + tt) * conv_dim + h * head_v + vbase + idx]);
    }
    __syncthreads();

    // ---- l2norm of q and k, and the scalar k.q, for all tb steps ----------
    // Warp w handles step w, so up to TB steps normalize concurrently.
    // Precision note: the reference applies l2norm to the BF16 q and k, BEFORE
    // casting to fp32, so the reduction result, the rsqrt and the product are
    // all rounded to 8 mantissa bits. The 1/sqrt(d) scale is applied in fp32
    // because the reference applies it after the cast.
    {
      const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
      constexpr int EPT = HEAD_K / 32;
      for (int tt = w; tt < tb; tt += THREADS / 32) {
        float* qp = sb_q + tt * HEAD_K;
        float* kp = sb_k + tt * HEAD_K;
        float qv[EPT], kv[EPT];
        float pq = 0.f, pk = 0.f;
        #pragma unroll
        for (int i = 0; i < EPT; ++i) {
          const int idx = i * 32 + lane;
          qv[i] = qp[idx]; kv[i] = kp[idx];
          pq = fmaf(qv[i], qv[i], pq);
          pk = fmaf(kv[i], kv[i], pk);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
          pq += __shfl_xor_sync(0xffffffffu, pq, o);
          pk += __shfl_xor_sync(0xffffffffu, pk, o);
        }
        pq = __bfloat162float(__float2bfloat16(pq));
        pk = __bfloat162float(__float2bfloat16(pk));
        const float nq = __bfloat162float(__float2bfloat16(rsqrtf(
                           __bfloat162float(__float2bfloat16(pq + 1e-6f)))));
        const float nk = __bfloat162float(__float2bfloat16(rsqrtf(
                           __bfloat162float(__float2bfloat16(pk + 1e-6f)))));
        const float dscale = rsqrtf(float(HEAD_K));
        float pkq = 0.f;
        #pragma unroll
        for (int i = 0; i < EPT; ++i) {
          const int idx = i * 32 + lane;
          const float qn = __bfloat162float(__float2bfloat16(qv[i] * nq)) * dscale;
          const float kn = __bfloat162float(__float2bfloat16(kv[i] * nk));
          qp[idx] = qn; kp[idx] = kn;
          pkq = fmaf(kn, qn, pkq);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) pkq += __shfl_xor_sync(0xffffffffu, pkq, o);
        if (lane == 0) sb_kq[tt] = pkq;
      }
    }
    __syncthreads();

    for (int tt = 0; tt < tb; ++tt) {
      const int t = t0 + tt;
      const float* s_q = sb_q + tt * HEAD_K;
      const float* s_k = sb_k + tt * HEAD_K;
      const float  s_kq = sb_kq[tt];
      const float g    = expf(gs[size_t(t) * num_v_heads + h]);
      const float beta = betas[size_t(t) * num_v_heads + h];

      // ---- one pass: A = S^T k, B = S^T q, tile kept in registers ---------
      float reg[KPT];
      float pa = 0.f, pb = 0.f;
      #pragma unroll 8
      for (int j = 0; j < KPT; ++j) {
        const float sv = S[(k0 + j) * SSTRIDE + v];
        reg[j] = sv;
        pa = fmaf(sv, s_k[k0 + j], pa);
        pb = fmaf(sv, s_q[k0 + j], pb);
      }
      red[kg * V_PER_BLOCK + v] = pa;
      red[KG * V_PER_BLOCK + kg * V_PER_BLOCK + v] = pb;
      __syncthreads();

      float A = 0.f, B = 0.f;
      #pragma unroll
      for (int j = 0; j < KG; ++j) {
        A += red[j * V_PER_BLOCK + v];
        B += red[KG * V_PER_BLOCK + j * V_PER_BLOCK + v];
      }
      const float delta = (sb_v[tt * V_PER_BLOCK + v] - g * A) * beta;
      if (kg == 0)
        out[size_t(t) * val_dim + h * head_v + vbase + v] =
            __float2bfloat16(g * B + delta * s_kq);

      // ---- writeback: S = g*S + k (outer) delta ---------------------------
      #pragma unroll 8
      for (int j = 0; j < KPT; ++j)
        S[(k0 + j) * SSTRIDE + v] = fmaf(g, reg[j], s_k[k0 + j] * delta);
      __syncthreads();
    }
  }

  for (int i = threadIdx.x; i < HEAD_K * V_PER_BLOCK; i += THREADS) {
    const int kk = i / V_PER_BLOCK, vv = i % V_PER_BLOCK;
    state[sbase + size_t(kk) * head_v + vv] = S[kk * SSTRIDE + vv];
  }
}

// ---------------------------------------------------------------- norm+gate
__global__ void k_norm_gate(__nv_bfloat16* __restrict__ out,
                            const __nv_bfloat16* __restrict__ x,
                            const __nv_bfloat16* __restrict__ z,
                            const __nv_bfloat16* __restrict__ w,
                            int rows, int D, float eps, int nh, int z_stride) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const size_t off = size_t(r) * D;
  // z lives at column offset z_off of a row that may be wider than nh*D
  const int tk = r / nh, hh = r % nh;
  const size_t zoff = size_t(tk) * z_stride + size_t(hh) * D;
  float acc = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = __bfloat162float(x[off + i]);
    acc += v * v;
  }
  #pragma unroll
  for (int s = 16; s > 0; s >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, s);
  __shared__ float part[8];
  const int nw = blockDim.x >> 5;
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = acc;
  __syncthreads();
  __shared__ float inv;
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < nw; ++i) t += part[i];
    inv = rsqrtf(t / D + eps);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    // The reference rounds to bf16 after the norm and BEFORE multiplying by the
    // weight, then gates in fp32. Reproduced exactly.
    const float n = __bfloat162float(__float2bfloat16(__bfloat162float(x[off + i]) * inv));
    const float wv = __bfloat162float(w[i]);
    out[off + i] = __float2bfloat16(wv * n * silu(__bfloat162float(z[zoff + i])));
  }
}

// Prefill conv: one thread per (channel, timestep). The decode kernel walks T
// serially per channel, which is right for T=1 but leaves 10240/256 = 40 blocks
// doing 4096 dependent iterations at prefill -- it measured 4.4 ms of a 15.7 ms
// layer. Here every output is independent; only the trailing state needs a
// second pass.
__global__ void k_conv_prefill(__nv_bfloat16* __restrict__ out,
                               const __nv_bfloat16* __restrict__ in,
                               const __nv_bfloat16* __restrict__ w,
                               const float* __restrict__ state,
                               int conv_dim, int K, int T, bool use_state, int in_stride) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(T) * conv_dim) return;
  const int t = int(i / conv_dim), c = int(i % conv_dim);
  float acc = 0.f;
  for (int j = 0; j < K; ++j) {
    const int tt = t + j - (K - 1);
    float x;
    if (tt >= 0)              x = __bfloat162float(in[size_t(tt) * in_stride + c]);
    else if (use_state)       x = state[size_t(c) * (K - 1) + (K - 1 + tt)];
    else                      x = 0.f;
    acc += __bfloat162float(w[size_t(c) * K + j]) * x;
  }
  out[i] = __float2bfloat16(silu(acc));
}

__global__ void k_conv_state_tail(float* __restrict__ state,
                                  const __nv_bfloat16* __restrict__ in,
                                  const float* __restrict__ old_state,
                                  int conv_dim, int K, int T, bool use_state, int in_stride) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(conv_dim) * (K - 1)) return;
  const int c = int(i / (K - 1)), j = int(i % (K - 1));
  // new state column j holds input at absolute position T - (K-1) + j
  const int tt = T - (K - 1) + j;
  if (tt >= 0)        state[i] = __bfloat162float(in[size_t(tt) * in_stride + c]);
  else if (use_state) state[i] = old_state[size_t(c) * (K - 1) + (K - 1 + tt)];
  else                state[i] = 0.f;
}

}  // namespace

void gdn_conv(__nv_bfloat16* out, float* state, const __nv_bfloat16* in,
              const __nv_bfloat16* w, int conv_dim, int K, int T, bool use_state,
              int in_stride, cudaStream_t st) {
  if (in_stride <= 0) in_stride = conv_dim;
  if (T == 1) {
    k_conv<<<(conv_dim + 255) / 256, 256, 0, st>>>(out, state, in, w, conv_dim, K, T,
                                                   use_state, in_stride);
    return;
  }
  // The tail must be read before the state is overwritten, so snapshot it.
  static float* saved = nullptr;
  static size_t saved_bytes = 0;
  const size_t need = size_t(conv_dim) * (K - 1) * sizeof(float);
  if (saved_bytes < need) {
    if (saved) cudaFree(saved);
    cudaMalloc(&saved, need);
    saved_bytes = need;
  }
  cudaMemcpyAsync(saved, state, need, cudaMemcpyDeviceToDevice, st);
  const size_t n = size_t(T) * conv_dim;
  k_conv_prefill<<<(n + 255) / 256, 256, 0, st>>>(out, in, w, saved, conv_dim, K, T, use_state, in_stride);
  const size_t m = size_t(conv_dim) * (K - 1);
  k_conv_state_tail<<<(m + 255) / 256, 256, 0, st>>>(state, in, saved, conv_dim, K, T, use_state, in_stride);
}

void gdn_gates(float* g, float* beta, const __nv_bfloat16* a, const __nv_bfloat16* b,
               const __nv_bfloat16* A_log, const __nv_bfloat16* dt_bias,
               int T, int H, cudaStream_t st) {
  k_gates<<<(T * H + 255) / 256, 256, 0, st>>>(g, beta, a, b, A_log, dt_bias, T, H);
}

// Splitting the v dimension across blocks is what breaks the 48-block ceiling.
// The recurrence is sequential in k but every v is INDEPENDENT -- A[v], delta[v],
// out[v] and the S[.][v] column all depend on v only, and the single shared
// quantity is the scalar k.q, which each block recomputes for ~128 flops. With
// the whole head in one block the state tile is 64 KB, pinning it to one block
// per SM and 8 warps; splitting v by 4 makes it 16 KB and lets several blocks
// share an SM.
#ifndef QWEN_GDN_THREADS
#define QWEN_GDN_THREADS 256
#endif
#ifndef QWEN_GDN_VPB
#define QWEN_GDN_VPB 16
#endif
#ifndef QWEN_GDN_TB
#define QWEN_GDN_TB 16
#endif

void gdn_scan(__nv_bfloat16* out, float* state, const __nv_bfloat16* qkv,
              const float* g, const float* beta, const GdnDims& d, int T,
              cudaStream_t st) {
  constexpr int THREADS = QWEN_GDN_THREADS;
  constexpr int VPB     = QWEN_GDN_VPB;
  const int v_blocks = d.head_v / VPB;
  const size_t sm = size_t(d.head_k) * (VPB + 1) * sizeof(float) +
                    size_t(2 * (THREADS / VPB)) * VPB * sizeof(float) + 32 * sizeof(float);
  (void)v_blocks;
  // Decode and prefill want different shapes. Decode is one step, so the extra
  // staging buffers only cost occupancy; prefill is 4096 sequential steps, where
  // batching timesteps and a wider v split both pay off. Instantiate both.
  static bool cfg = false;
  if (!cfg) {
    // The 99 KB opt-in ceiling covers static AND dynamic shared memory, and this
    // kernel already holds ~1.5 KB statically. Asking for the full ceiling makes
    // cudaFuncSetAttribute fail silently, leaving the 48 KB default in place and
    // turning the launch into "invalid argument" -- which reads like a grid/block
    // bug and is not one.
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    auto opt_in = [&](auto fn) {
      cudaFuncAttributes fa{};
      cudaFuncGetAttributes(&fa, fn);
      const int want = optin - int(fa.sharedSizeBytes);
      const cudaError_t e = cudaFuncSetAttribute(
          fn, cudaFuncAttributeMaxDynamicSharedMemorySize, want);
      if (e != cudaSuccess || sm > size_t(want)) {
        fprintf(stderr, "gdn_scan: shared memory opt-in failed (%s) (%zu needed, %d available)\n",
                cudaGetErrorString(e),
                sm, want);
        abort();
      }
    };
    opt_in(k_scan<128, VPB, THREADS, 1>);
    opt_in(k_scan<128, VPB, THREADS, QWEN_GDN_TB>);
    cfg = true;
  }
  if (T <= 8)
    k_scan<128, VPB, THREADS, 1><<<d.num_v_heads * v_blocks, THREADS, sm, st>>>(
        out, state, qkv, g, beta, T, d.conv_dim(), d.key_dim(), d.val_dim(),
        d.num_v_heads, d.hk_div(), d.hk_mod(), d.head_v, v_blocks);
  else
    k_scan<128, VPB, THREADS, QWEN_GDN_TB><<<d.num_v_heads * v_blocks, THREADS, sm, st>>>(
        out, state, qkv, g, beta, T, d.conv_dim(), d.key_dim(), d.val_dim(),
        d.num_v_heads, d.hk_div(), d.hk_mod(), d.head_v, v_blocks);
}

void gdn_norm_gate(__nv_bfloat16* out, const __nv_bfloat16* x, const __nv_bfloat16* z,
                   const __nv_bfloat16* w, int T, int H, int D, float eps,
                   int z_stride, cudaStream_t st) {
  if (z_stride <= 0) z_stride = H * D;
  k_norm_gate<<<T * H, 128, 0, st>>>(out, x, z, w, T * H, D, eps, H, z_stride);
}

}  // namespace qwen
