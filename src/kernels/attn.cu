#include "attn.cuh"
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cuda_runtime.h>

namespace qwen {

// cuBLAS returns NOT_SUPPORTED for unsupported dtype combinations rather than
// doing anything, so an unchecked call silently produces zeros.
#define CBA(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS %s:%d status %d\n",__FILE__,__LINE__,int(s_)); abort(); } } while(0)

namespace {

// ---------------------------------------------------------------- fp8 e4m3
// sm_86 has no FP8 hardware, so these are pure bit manipulation. E4M3 is
// 1-4-3 with bias 7 and no infinities; exponent 0 is subnormal.
__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
  const uint32_t x = b;
  const uint32_t e = (x >> 3) & 0xFu;
  const uint32_t m = x & 0x7u;
  const uint32_t bits = ((e + 120u) << 23) | (m << 20);   // 120 = 127 - 7
  const float vn = __int_as_float(bits);
  const float vs = float(m) * (1.0f / 512.0f);            // m * 2^-9
  const float v  = (e == 0u) ? vs : vn;
  return (x & 0x80u) ? -v : v;
}

__device__ __forceinline__ uint8_t f32_to_e4m3(float f) {
  const uint32_t xb = __float_as_uint(f);
  const uint32_t sgn = (xb >> 31) << 7;
  float a = fabsf(f);
  if (!(a > 0.f)) return uint8_t(sgn);                    // zero or NaN-as-zero
  a = fminf(a, 448.0f);                                   // e4m3 max, saturating
  const uint32_t ab = __float_as_uint(a);
  int e = int((ab >> 23) & 0xFFu) - 127;
  uint32_t m = ab & 0x7FFFFFu;
  if (e < -6) {                                           // subnormal
    const float scaled = a * 512.0f;                      // units of 2^-9
    uint32_t q = uint32_t(scaled + 0.5f);
    // round-to-nearest-even on the tie
    if (scaled + 0.5f == floorf(scaled + 0.5f) && (q & 1u)) --q;
    if (q > 7u) return uint8_t(sgn | (1u << 3));           // rounded up into normal
    return uint8_t(sgn | q);
  }
  // round mantissa to 3 bits, nearest-even
  const uint32_t round = ((m >> 20) & 1u) + 0x7FFFFu;
  m += round;
  if (m & 0x800000u) { m = 0; ++e; }
  if (e > 8) return uint8_t(sgn | 0x7Eu);                  // clamp to max finite
  uint32_t mm = m >> 20;
  if (e == 8 && mm > 6u) mm = 6u;   // exp=15,mant=7 is NaN in e4m3, not a value
  return uint8_t(sgn | (uint32_t(e + 7) << 3) | mm);
}

// ---------------------------------------------------------------- rope
__global__ void k_rope_tables_dev(float* __restrict__ cosv, float* __restrict__ sinv,
                                  const int32_t* __restrict__ dpos, int half, float theta) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= half) return;
  const int32_t p = dpos[0];
  const float inv = float(exp(-log(double(theta)) * (double(2 * j) / double(2 * half))));
  const float ang = float(p) * inv;
  cosv[j] = __bfloat162float(__float2bfloat16(cosf(ang)));
  sinv[j] = __bfloat162float(__float2bfloat16(sinf(ang)));
}

__global__ void k_rope_tables(float* __restrict__ cosv, float* __restrict__ sinv,
                              const int32_t* __restrict__ pt,
                              const int32_t* __restrict__ ph,
                              const int32_t* __restrict__ pw,
                              int T, int half, float theta) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * half) return;
  const int t = i / half, j = i % half;
  // mrope: frequency j takes its position from T, H or W by j % 3, which is
  // exactly sections [11,11,10] over 32 frequencies. Text-only passes the same
  // array three times, making this an identity -- deliberately not special-cased.
  const int32_t p = (j % 3 == 0) ? pt[t] : (j % 3 == 1) ? ph[t] : pw[t];
  // inv_freq MUST be computed accurately. __powf carries ~1e-4 relative error,
  // and at position 131071 that is 13 RADIANS of angle error -- the output stays
  // fluent at short context and falls apart at long, which is the directive's
  // failure mode #1. Computed in double, once per frequency.
  // inv_freq is computed in double and rounded to float, because torch stores a
  // float32 inv_freq; the PRODUCT is then taken in float32, because torch does
  // `inv_freq.float() @ position_ids.float()`. Doing the product in double and
  // rounding once is more accurate and lands one bf16 ulp away at position
  // 131071 -- close, but not the same trajectory.
  const float inv = float(exp(-log(double(theta)) * (double(2 * j) / double(2 * half))));
  const float ang = float(p) * inv;
  // sinf/cosf, not __sinf/__cosf: the fast intrinsics lose accuracy well before
  // an argument of 131071 radians, and they do it silently.
  // The reference returns cos/sin cast to the activation dtype (bf16) before the
  // rotation, so the table carries 8 mantissa bits. Rounding here keeps us on
  // the same trajectory as HF rather than being gratuitously more precise.
  cosv[i] = __bfloat162float(__float2bfloat16(cosf(ang)));
  sinv[i] = __bfloat162float(__float2bfloat16(sinf(ang)));
}

// ---------------------------------------------------------------- prepare
// One block per (token, q-or-kv head). 256 threads, one per head dim.
__global__ void k_prepare_q(__nv_bfloat16* __restrict__ q_out,
                            __nv_bfloat16* __restrict__ gate,
                            const __nv_bfloat16* __restrict__ qkv_in,
                            const __nv_bfloat16* __restrict__ qn_w,
                            const float* __restrict__ cosv, const float* __restrict__ sinv,
                            int T, int nq, int D, int rot, float eps, int stride_in) {
  const int t = blockIdx.x, h = blockIdx.y;
  const int d = threadIdx.x;
  // q_proj packs [query(D) | gate(D)] per head
  const size_t base = size_t(t) * stride_in + size_t(h) * 2 * D;
  const float qv = __bfloat162float(qkv_in[base + d]);
  const float gv = __bfloat162float(qkv_in[base + D + d]);
  gate[size_t(t) * nq * D + size_t(h) * D + d] = __float2bfloat16(gv);

  __shared__ float red[8];
  float s = qv * qv;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  if ((d & 31) == 0) red[d >> 5] = s;
  __syncthreads();
  __shared__ float inv;
  if (d == 0) { float a = 0.f; for (int i = 0; i < 8; ++i) a += red[i]; inv = rsqrtf(a / D + eps); }
  __syncthreads();

  // Qwen3_5RMSNorm is  norm(x.float()) * (1.0 + weight.float()), kept in fp32
  // until the final cast -- NOT the Llama-style w * bf16(norm(x)), and NOT the
  // gated variant GDN uses. The weight is initialised to zeros, so the "1.0 +"
  // is load-bearing rather than cosmetic.
  float x = (qv * inv) * (1.0f + __bfloat162float(qn_w[d]));
  // rotate the first `rot` dims only; the remaining D-rot pass through
  __shared__ float sh[256];
  sh[d] = x;
  __syncthreads();
  if (d < rot) {
    const int half = rot / 2;
    const int j = d % half;
    const float c = cosv[size_t(t) * half + j], sn = sinv[size_t(t) * half + j];
    x = (d < half) ? (sh[d] * c - sh[d + half] * sn)
                   : (sh[d] * c + sh[d - half] * sn);
  }
  q_out[size_t(t) * nq * D + size_t(h) * D + d] = __float2bfloat16(x);
}

__global__ void k_prepare_kv(uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                             const __nv_bfloat16* __restrict__ qkv_in,
                             const __nv_bfloat16* __restrict__ kn_w,
                             const float* __restrict__ cosv, const float* __restrict__ sinv,
                             int T, int nkv, int D, int rot, float eps,
                             int stride_in, int k_off, int v_off,
                             int cache_pos, int max_ctx, const int32_t* dpos) {
  const int t = blockIdx.x, h = blockIdx.y;
  const int d = threadIdx.x;
  const size_t base = size_t(t) * stride_in;
  const float kv = __bfloat162float(qkv_in[base + k_off + size_t(h) * D + d]);
  const float vv = __bfloat162float(qkv_in[base + v_off + size_t(h) * D + d]);

  __shared__ float red[8];
  float s = kv * kv;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  if ((d & 31) == 0) red[d >> 5] = s;
  __syncthreads();
  __shared__ float inv;
  if (d == 0) { float a = 0.f; for (int i = 0; i < 8; ++i) a += red[i]; inv = rsqrtf(a / D + eps); }
  __syncthreads();

  float x = (kv * inv) * (1.0f + __bfloat162float(kn_w[d]));
  __shared__ float sh[256];
  sh[d] = x;
  __syncthreads();
  if (d < rot) {
    const int half = rot / 2;
    const int j = d % half;
    const float c = cosv[size_t(t) * half + j], sn = sinv[size_t(t) * half + j];
    x = (d < half) ? (sh[d] * c - sh[d + half] * sn)
                   : (sh[d] * c + sh[d - half] * sn);
  }
  const int cp = dpos ? dpos[0] : cache_pos;
  const size_t off = (size_t(cp + t) * nkv + h) * D + d;
  kc[off] = f32_to_e4m3(x);
  vc[off] = f32_to_e4m3(vv);
}

// ---------------------------------------------------------------- decode
// Block owns one KV head and one KV split. All q_per_kv query heads that share
// that KV head live in the same block, so K and V are read ONCE rather than
// once per query head -- at 6 query heads per KV head that would otherwise be
// 6x the KV traffic, and KV traffic is 4 GiB per token at 128K context.
#ifndef QWEN_ATTN_WARPS
#define QWEN_ATTN_WARPS 4
#endif
constexpr int DEC_WARPS = QWEN_ATTN_WARPS;
constexpr int DEC_THREADS = DEC_WARPS * 32;
constexpr int DPL = 8;              // head dims per lane: 32 lanes * 8 = 256

template <int QPK>
__global__ __launch_bounds__(DEC_THREADS) void k_attn_decode(
    float* __restrict__ part_o, float* __restrict__ part_m, float* __restrict__ part_l,
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    int ctx_len, int nkv, int D, int nq, float scaling, int splits,
    const int32_t* __restrict__ d_ctx) {
  const int kvh = blockIdx.x, sp = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int d0 = lane * DPL;

  if (d_ctx) ctx_len = d_ctx[0];
  const int per = (ctx_len + splits - 1) / splits;
  const int p0 = sp * per, p1 = min(p0 + per, ctx_len);

  // e4m3 -> float costs ~7 ALU ops per value, and this loop does 16 of them per
  // lane per KV position. Measured: stubbing the conversion out took 15.15 ms
  // to 10.17 ms per token at 128K context, so it was a third of the kernel.
  // A 1 KB shared LUT replaces it with one load. Bank conflicts are mild because
  // e4m3 byte values across a warp are effectively random.
  __shared__ float lut[256];
  for (int i = threadIdx.x; i < 256; i += DEC_THREADS) lut[i] = e4m3_to_f32(uint8_t(i));
  __syncthreads();

  float qr[QPK][DPL], acc[QPK][DPL], m[QPK], l[QPK];
  #pragma unroll
  for (int j = 0; j < QPK; ++j) {
    const int h = kvh * QPK + j;
    #pragma unroll
    for (int i = 0; i < DPL; ++i) {
      qr[h < nq ? j : j][i] = __bfloat162float(q[size_t(h) * D + d0 + i]);
      acc[j][i] = 0.f;
    }
    m[j] = -FLT_MAX; l[j] = 0.f;
  }

  for (int p = p0 + warp; p < p1; p += DEC_WARPS) {
    // ONE 8-byte load per lane, not eight 1-byte loads. A warp then covers 256
    // contiguous bytes in a single transaction. Byte-at-a-time addressing
    // measured 308 GB/s of KV bandwidth; at 128K context that is 13.9 ms per
    // token against a 16.5 ms weight budget, i.e. it would have halved decode
    // speed at long context.
    const uint2 kw = *reinterpret_cast<const uint2*>(kc + (size_t(p) * nkv + kvh) * D + d0);
    const uint2 vw = *reinterpret_cast<const uint2*>(vc + (size_t(p) * nkv + kvh) * D + d0);
    float kv[DPL], vv[DPL];
    #pragma unroll
    for (int i = 0; i < DPL; ++i) {
      const uint32_t kb = (i < 4) ? kw.x : kw.y;
      const uint32_t vb = (i < 4) ? vw.x : vw.y;
      const int sh = (i & 3) * 8;
      kv[i] = lut[(kb >> sh) & 0xFFu];
      vv[i] = lut[(vb >> sh) & 0xFFu];
    }

    #pragma unroll
    for (int j = 0; j < QPK; ++j) {
      float s = 0.f;
      #pragma unroll
      for (int i = 0; i < DPL; ++i) s = fmaf(qr[j][i], kv[i], s);
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
      s *= scaling;
      const float mn = fmaxf(m[j], s);
      const float corr = __expf(m[j] - mn);
      const float pw = __expf(s - mn);
      l[j] = l[j] * corr + pw;
      #pragma unroll
      for (int i = 0; i < DPL; ++i) acc[j][i] = fmaf(acc[j][i], corr, pw * vv[i]);
      m[j] = mn;
    }
  }

  // combine the DEC_WARPS partial softmaxes within the block
  extern __shared__ float sm[];
  float* s_o = sm;                                     // [DEC_WARPS][QPK][256]
  float* s_m = sm + DEC_WARPS * QPK * 256;             // [DEC_WARPS][QPK]
  float* s_l = s_m + DEC_WARPS * QPK;
  #pragma unroll
  for (int j = 0; j < QPK; ++j) {
    #pragma unroll
    for (int i = 0; i < DPL; ++i) s_o[((warp * QPK) + j) * 256 + d0 + i] = acc[j][i];
    if (lane == 0) { s_m[warp * QPK + j] = m[j]; s_l[warp * QPK + j] = l[j]; }
  }
  __syncthreads();

  if (warp == 0) {
    #pragma unroll
    for (int j = 0; j < QPK; ++j) {
      float mm = -FLT_MAX;
      for (int w = 0; w < DEC_WARPS; ++w) mm = fmaxf(mm, s_m[w * QPK + j]);
      float ll = 0.f, o[DPL];
      #pragma unroll
      for (int i = 0; i < DPL; ++i) o[i] = 0.f;
      for (int w = 0; w < DEC_WARPS; ++w) {
        const float c = __expf(s_m[w * QPK + j] - mm);
        ll += s_l[w * QPK + j] * c;
        #pragma unroll
        for (int i = 0; i < DPL; ++i) o[i] = fmaf(s_o[((w * QPK) + j) * 256 + d0 + i], c, o[i]);
      }
      const int h = kvh * QPK + j;
      #pragma unroll
      for (int i = 0; i < DPL; ++i)
        part_o[(size_t(sp) * nq + h) * 256 + d0 + i] = o[i];
      if (lane == 0) {
        part_m[size_t(sp) * nq + h] = mm;
        part_l[size_t(sp) * nq + h] = ll;
      }
    }
  }
}

__global__ void k_decode_combine(__nv_bfloat16* __restrict__ out,
                                 const float* __restrict__ part_o,
                                 const float* __restrict__ part_m,
                                 const float* __restrict__ part_l,
                                 int nq, int D, int splits) {
  const int h = blockIdx.x, d = threadIdx.x;
  __shared__ float mm, ll;
  if (d == 0) {
    float m = -FLT_MAX, l = 0.f;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, part_m[size_t(s) * nq + h]);
    for (int s = 0; s < splits; ++s) l += part_l[size_t(s) * nq + h] * __expf(part_m[size_t(s) * nq + h] - m);
    mm = m; ll = l;
  }
  __syncthreads();
  float o = 0.f;
  // Fixed split order, so greedy decode stays bitwise reproducible.
  for (int s = 0; s < splits; ++s)
    o = fmaf(part_o[(size_t(s) * nq + h) * D + d], __expf(part_m[size_t(s) * nq + h] - mm), o);
  out[size_t(h) * D + d] = __float2bfloat16(o / ll);
}

// ---------------------------------------------------------------- gate
__global__ void k_gate(__nv_bfloat16* __restrict__ o, const __nv_bfloat16* __restrict__ g, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float gv = __bfloat162float(g[i]);
  o[i] = __float2bfloat16(__bfloat162float(o[i]) / (1.f + expf(-gv)));
}

// ---------------------------------------------------------------- prefill
// Dequantize an FP8 KV tile to bf16 so cuBLAS can consume it.
__global__ void k_fill(float* __restrict__ p, float v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

__global__ void k_kv_deq(__nv_bfloat16* __restrict__ dst, const uint8_t* __restrict__ src,
                         int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(e4m3_to_f32(src[i]));
}

// Causal-masked online-softmax update over one KV tile.
// scores: [nq][q_tile][kv_tile] fp32 in, probabilities out.
__global__ void k_softmax_tile(float* __restrict__ scores,
                               __nv_bfloat16* __restrict__ probs,
                               float* __restrict__ mrun,
                               float* __restrict__ lrun, float* __restrict__ corr,
                               int q_cap, int q_n, int kv_cap, int kv_tile,
                               int kv0, int q_abs0, float scaling, int ctx_len) {
  // blockIdx.x enumerates the LIVE rows (nq * q_n) but the buffers are strided
  // by the tile CAPACITY q_cap, so the two must be decoded separately. Deriving
  // h from blockIdx.x / q_cap silently addressed head 0 for every partial tile.
  const int h = blockIdx.x / q_n, qi = blockIdx.x % q_n;
  const int row = h * q_cap + qi;
  // Row stride is the tile CAPACITY, not the live width. Both this and the
  // GEMM's ldc have to use it, and they only coincide on a full tile -- which
  // no context under 2048 ever produces, so a short prompt read the wrong row.
  float* s = scores + size_t(row) * kv_cap;
  const int qpos = q_abs0 + qi;

  float mx = -FLT_MAX;
  for (int i = threadIdx.x; i < kv_tile; i += blockDim.x) {
    const int kpos = kv0 + i;
    float v = (kpos <= qpos && kpos < ctx_len) ? s[i] * scaling : -FLT_MAX;
    s[i] = v;
    mx = fmaxf(mx, v);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  __shared__ float rm[8], rl[8];
  const int nw = blockDim.x >> 5;
  if ((threadIdx.x & 31) == 0) rm[threadIdx.x >> 5] = mx;
  __syncthreads();
  __shared__ float mtile;
  if (threadIdx.x == 0) { float a = -FLT_MAX; for (int i = 0; i < nw; ++i) a = fmaxf(a, rm[i]); mtile = a; }
  __syncthreads();

  float sum = 0.f;
  for (int i = threadIdx.x; i < kv_tile; i += blockDim.x) {
    const float e = (s[i] == -FLT_MAX) ? 0.f : __expf(s[i] - mtile);
    s[i] = e;
    sum += e;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if ((threadIdx.x & 31) == 0) rl[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = 0.f; for (int i = 0; i < nw; ++i) a += rl[i];
    const float mprev = mrun[row];
    const float mnew = fmaxf(mprev, mtile);
    const float c = (mprev == -FLT_MAX) ? 0.f : __expf(mprev - mnew);
    corr[row] = c;
    lrun[row] = lrun[row] * c + a * __expf(mtile - mnew);
    mrun[row] = mnew;
    rm[0] = __expf(mtile - mnew);
  }
  __syncthreads();
  const float pscale = rm[0];
  __nv_bfloat16* pb = probs + size_t(row) * kv_cap;
  for (int i = threadIdx.x; i < kv_tile; i += blockDim.x) {
    s[i] *= pscale;
    pb[i] = __float2bfloat16(s[i]);
  }
}

__global__ void k_acc_tile(float* __restrict__ o, const float* __restrict__ corr,
                           int q_cap, int q_n, int D) {
  const int h = blockIdx.x / q_n, qi = blockIdx.x % q_n, d = threadIdx.x;
  const int row = h * q_cap + qi;
  if (d < D) o[size_t(row) * D + d] *= corr[row];
}

__global__ void k_finish(__nv_bfloat16* __restrict__ out, const float* __restrict__ o,
                         const float* __restrict__ lrun, int q_cap, int q_n, int nq,
                         int D, int q_off) {
  const int h = blockIdx.x / q_n, qi = blockIdx.x % q_n, d = threadIdx.x;
  const int row = h * q_cap + qi;
  out[(size_t(q_off + qi) * nq + h) * D + d] = __float2bfloat16(o[size_t(row) * D + d] / lrun[row]);
}

}  // namespace

// ================================================================ host
void rope_tables(float* c, float* s, const int32_t* pt, const int32_t* ph,
                 const int32_t* pw, int T, int rot, float theta, cudaStream_t st) {
  const int half = rot / 2;
  k_rope_tables<<<(T * half + 255) / 256, 256, 0, st>>>(c, s, pt, ph, pw, T, half, theta);
}

void attn_prepare(__nv_bfloat16* q_out, __nv_bfloat16* gate, uint8_t* kc, uint8_t* vc,
                  const __nv_bfloat16* qkv_in, const __nv_bfloat16* qnw,
                  const __nv_bfloat16* knw, const float* cosv, const float* sinv,
                  int T, int cache_pos, int max_ctx, const AttnDims& d, cudaStream_t st) {
  const int stride = d.q_proj_out() + 2 * d.kv_proj_out();
  k_prepare_q<<<dim3(T, d.num_q_heads), d.head_dim, 0, st>>>(
      q_out, gate, qkv_in, qnw, cosv, sinv, T, d.num_q_heads, d.head_dim,
      d.rotary_dim, d.rms_eps, stride);
  k_prepare_kv<<<dim3(T, d.num_kv_heads), d.head_dim, 0, st>>>(
      kc, vc, qkv_in, knw, cosv, sinv, T, d.num_kv_heads, d.head_dim, d.rotary_dim,
      d.rms_eps, stride, d.q_proj_out(), d.q_proj_out() + d.kv_proj_out(),
      cache_pos, max_ctx, nullptr);
}

int attn_decode_splits(int ctx_len) {
  static int sms = 0;
  if (!sms) cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  // 4 KV heads only, so without splitting there are 4 blocks. Split until the
  // machine is full, but keep at least 256 positions per split so the per-split
  // combine does not dominate.
  // Split count swept over 1/2/4/8 waves at 64K and 128K; 8 is the knee.
  static int mult = 0;
  if (!mult) { const char* e = getenv("QWEN_ATTN_WAVES"); mult = e ? atoi(e) : 8; }
  int s = (sms * mult) / 4;
  const int maxs = (ctx_len + 255) / 256;
  if (s > maxs) s = maxs;
  return s < 1 ? 1 : s;
}

size_t attn_decode_workspace_bytes(const AttnDims& d, int max_splits) {
  return size_t(max_splits) * d.num_q_heads * (d.head_dim + 2) * sizeof(float);
}

void attn_decode(__nv_bfloat16* out, const __nv_bfloat16* q, const uint8_t* kc,
                 const uint8_t* vc, int ctx_len, int max_ctx, const AttnDims& d,
                 float* ws, int splits, cudaStream_t st) {
  float* po = ws;
  float* pm = ws + size_t(splits) * d.num_q_heads * d.head_dim;
  float* pl = pm + size_t(splits) * d.num_q_heads;
  const size_t sm = size_t(DEC_WARPS) * d.q_per_kv() * d.head_dim * sizeof(float) +
                    size_t(2 * DEC_WARPS) * d.q_per_kv() * sizeof(float);
  static bool cfg = false;
  if (!cfg) {
    int optin = 0;
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    cudaFuncAttributes fa{};
    cudaFuncGetAttributes(&fa, k_attn_decode<6>);
    cudaFuncSetAttribute(k_attn_decode<6>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         optin - int(fa.sharedSizeBytes));
    cfg = true;
  }
  k_attn_decode<6><<<dim3(d.num_kv_heads, splits), DEC_THREADS, sm, st>>>(
      po, pm, pl, q, kc, vc, ctx_len, d.num_kv_heads, d.head_dim, d.num_q_heads,
      d.scaling(), splits, nullptr);
  k_decode_combine<<<d.num_q_heads, d.head_dim, 0, st>>>(out, po, pm, pl,
                                                         d.num_q_heads, d.head_dim, splits);
}

void rope_tables_dev(float* c, float* s, const int32_t* dpos, int rot, float theta,
                     cudaStream_t st) {
  const int half = rot / 2;
  k_rope_tables_dev<<<(half + 63) / 64, 64, 0, st>>>(c, s, dpos, half, theta);
}

void attn_prepare_dev(__nv_bfloat16* q_out, __nv_bfloat16* gate, uint8_t* kc, uint8_t* vc,
                      const __nv_bfloat16* qkv_in, const __nv_bfloat16* qnw,
                      const __nv_bfloat16* knw, const float* cosv, const float* sinv,
                      const int32_t* dpos, int max_ctx, const AttnDims& d,
                      cudaStream_t st) {
  const int stride = d.q_proj_out() + 2 * d.kv_proj_out();
  k_prepare_q<<<dim3(1, d.num_q_heads), d.head_dim, 0, st>>>(
      q_out, gate, qkv_in, qnw, cosv, sinv, 1, d.num_q_heads, d.head_dim,
      d.rotary_dim, d.rms_eps, stride);
  k_prepare_kv<<<dim3(1, d.num_kv_heads), d.head_dim, 0, st>>>(
      kc, vc, qkv_in, knw, cosv, sinv, 1, d.num_kv_heads, d.head_dim, d.rotary_dim,
      d.rms_eps, stride, d.q_proj_out(), d.q_proj_out() + d.kv_proj_out(),
      0, max_ctx, dpos);
}

void attn_decode_dev(__nv_bfloat16* out, const __nv_bfloat16* q, const uint8_t* kc,
                     const uint8_t* vc, const int32_t* d_ctx, int max_ctx,
                     const AttnDims& d, float* ws, int splits, cudaStream_t st) {
  float* po = ws;
  float* pm = ws + size_t(splits) * d.num_q_heads * d.head_dim;
  float* pl = pm + size_t(splits) * d.num_q_heads;
  const size_t sm = size_t(DEC_WARPS) * d.q_per_kv() * d.head_dim * sizeof(float) +
                    size_t(2 * DEC_WARPS) * d.q_per_kv() * sizeof(float);
  k_attn_decode<6><<<dim3(d.num_kv_heads, splits), DEC_THREADS, sm, st>>>(
      po, pm, pl, q, kc, vc, 0, d.num_kv_heads, d.head_dim, d.num_q_heads,
      d.scaling(), splits, d_ctx);
  k_decode_combine<<<d.num_q_heads, d.head_dim, 0, st>>>(out, po, pm, pl,
                                                         d.num_q_heads, d.head_dim, splits);
}

void attn_output_gate(__nv_bfloat16* o, const __nv_bfloat16* g, int n, cudaStream_t st) {
  k_gate<<<(n + 255) / 256, 256, 0, st>>>(o, g, n);
}

size_t attn_prefill_scratch_bytes(const AttnDims& d, int kv_tile, int q_tile) {
  return size_t(d.num_q_heads) * q_tile * kv_tile * sizeof(float);
}

void attn_prefill(__nv_bfloat16* out, const __nv_bfloat16* q, const uint8_t* kc,
                  const uint8_t* vc, int T, int ctx_len, int q_offset, int max_ctx,
                  const AttnDims& d, __nv_bfloat16* kv_scratch, float* score_scratch,
                  cublasHandle_t cublas, cudaStream_t st) {
  // Attention prefill runs as tiled cuBLAS GEMMs with an online softmax between
  // them, rather than a hand-written flash kernel. Prefill is dominated by the
  // projection GEMMs (~2.9 s at 4096 tokens); attention is ~3.3 TFLOP against
  // their 199, so what matters is reaching tensor-core speed, not saving the
  // extra score-matrix traffic. Tiling over KV keeps the score matrix bounded
  // regardless of context length.
  const int KVT = 2048, QT = 128, D = d.head_dim, NQ = d.num_q_heads;
  const int nkvh = d.num_kv_heads, qpk = d.q_per_kv();

  float *o_acc, *mrun, *lrun, *corr;
  __nv_bfloat16* probs = nullptr;
  cudaMallocAsync(&o_acc, size_t(NQ) * QT * D * sizeof(float), st);
  cudaMallocAsync(&mrun,  size_t(NQ) * QT * sizeof(float), st);
  cudaMallocAsync(&lrun,  size_t(NQ) * QT * sizeof(float), st);
  cudaMallocAsync(&corr,  size_t(NQ) * QT * sizeof(float), st);
  cudaMallocAsync(&probs, size_t(NQ) * QT * KVT * sizeof(__nv_bfloat16), st);

  cublasSetStream(cublas, st);
  const float one = 1.f, zero = 0.f;

  for (int q0 = 0; q0 < T; q0 += QT) {
    const int qn = min(QT, T - q0);
    cudaMemsetAsync(o_acc, 0, size_t(NQ) * QT * D * sizeof(float), st);
    cudaMemsetAsync(lrun, 0, size_t(NQ) * QT * sizeof(float), st);
    // mrun = -FLT_MAX. memset(0xFF) gives 0xFFFFFFFF, which is NaN, not
    // -FLT_MAX; fmaxf then quietly drops it but the (mprev == -FLT_MAX) guard
    // does not fire, so every running max became NaN and every output NaN.
    k_fill<<<(NQ * QT + 255) / 256, 256, 0, st>>>(mrun, -FLT_MAX, NQ * QT);

    const int qmax = q_offset + q0 + qn - 1;
    for (int kv0 = 0; kv0 < ctx_len; kv0 += KVT) {
      if (kv0 > qmax) break;                       // fully masked by causality
      const int kvn = min(KVT, ctx_len - kv0);

      // dequantize this KV tile once for all heads
      const size_t tile_elems = size_t(kvn) * nkvh * D;
      k_kv_deq<<<(tile_elems + 255) / 256, 256, 0, st>>>(
          kv_scratch, kc + size_t(kv0) * nkvh * D, int(tile_elems));
      __nv_bfloat16* vbuf = kv_scratch + tile_elems;
      k_kv_deq<<<(tile_elems + 255) / 256, 256, 0, st>>>(
          vbuf, vc + size_t(kv0) * nkvh * D, int(tile_elems));

      // S[h] = Q[h] (qn x D) @ K[h]^T (D x kvn), one GEMM per query head so the
      // 6:1 GQA sharing costs no extra memory
      for (int h = 0; h < NQ; ++h) {
        const int kh = h / qpk;
        CBA(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, kvn, qn, D,
                     &one,
                     kv_scratch + size_t(kh) * D, CUDA_R_16BF, nkvh * D,
                     q + (size_t(q0) * NQ + h) * D, CUDA_R_16BF, NQ * D,
                     &zero,
                     // ldc is the TILE STRIDE, not the live tile width: the
                     // softmax indexes rows at KVT and the two only coincide on
                     // a full tile, which no context under 2048 ever produces.
                     score_scratch + size_t(h) * QT * KVT, CUDA_R_32F, KVT,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      k_softmax_tile<<<NQ * qn, 256, 0, st>>>(score_scratch, probs, mrun, lrun, corr,
                                              QT, qn, KVT, kvn, kv0, q_offset + q0,
                                              d.scaling(), ctx_len);
      k_acc_tile<<<NQ * qn, D, 0, st>>>(o_acc, corr, QT, qn, D);
      for (int h = 0; h < NQ; ++h) {
        const int kh = h / qpk;
        CBA(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, D, qn, kvn,
                     &one,
                     vbuf + size_t(kh) * D, CUDA_R_16BF, nkvh * D,
                     probs + size_t(h) * QT * KVT, CUDA_R_16BF, KVT,
                     &one,
                     o_acc + size_t(h) * QT * D, CUDA_R_32F, D,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
    }
    k_finish<<<NQ * qn, D, 0, st>>>(out, o_acc, lrun, QT, qn, NQ, D, q0);
  }
  cudaFreeAsync(o_acc, st); cudaFreeAsync(mrun, st);
  cudaFreeAsync(lrun, st);  cudaFreeAsync(corr, st);
  cudaFreeAsync(probs, st);
}

}  // namespace qwen
