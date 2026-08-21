#include "gemv_w4a16.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace qwen {
namespace {

constexpr int WARP  = 32;
#ifndef QWEN_GEMV_WARPS
#define QWEN_GEMV_WARPS 1
#endif
constexpr int WARPS = QWEN_GEMV_WARPS;
constexpr int BLOCK = WARP * WARPS;
constexpr int ROWS_PER_BLOCK = WARP * WARPS;

// bf16 0x4300 is exactly 128.0 and its ulp there is 1, so (0x4300 | n) is the
// exact value 128 + n for n in [0,15].
constexpr uint32_t kMagic = 0x43004300u;
constexpr uint32_t kMask  = 0x000F000Fu;

__device__ __forceinline__ float2 nib2(uint32_t w, int shift) {
  const uint32_t t = ((w >> shift) & kMask) | kMagic;
  __nv_bfloat162 b;
  *reinterpret_cast<uint32_t*>(&b) = t;
  return __bfloat1622float2(b);       // (128 + nibble_lo, 128 + nibble_hi)
}

// ---------------------------------------------------------------- repack
// [out][in/8] -> [out/32][in/8][32], and the same 32-row interleave for scale
// and zero point. The zero point also moves from its packed [out/8][G] u32 form
// to one byte per (row, group): 4-bit packing would save 0.09 GiB per token
// (0.7%) but costs a shift+mask on every group in the inner loop, and the
// zero-point stream is already the smallest of the three.
__global__ void k_repack(uint32_t* __restrict__ dq, __nv_bfloat16* __restrict__ ds,
                         uint8_t* __restrict__ dz,
                         const uint32_t* __restrict__ sq, const __nv_bfloat16* __restrict__ ss,
                         const uint32_t* __restrict__ sz,
                         int out_f, int words_per_row, int G, int row_offset) {
  const int so = blockIdx.x * blockDim.y + threadIdx.y;   // source row
  if (so >= out_f) return;
  const int o = so + row_offset;                          // destination row
  const int b = o >> 5, l = o & 31;

  for (int t = threadIdx.x; t < words_per_row; t += blockDim.x)
    dq[(size_t(b) * words_per_row + t) * 32 + l] = sq[size_t(so) * words_per_row + t];

  for (int g = threadIdx.x; g < G; g += blockDim.x) {
    ds[(size_t(b) * G + g) * 32 + l] = ss[size_t(so) * G + g];
    const uint32_t zw = sz[(size_t(so) >> 3) * G + g];
    dz[(size_t(b) * G + g) * 32 + l] = uint8_t((zw >> ((so & 7) * 4)) & 0xFu);
  }
}

// ---------------------------------------------------------------- prologue
// Widen x to fp32 once (so the inner loop never converts) and take per-group
// sums, which the zero-point correction needs at group granularity rather than
// per weight.
// One block per group: widen that group's activations to fp32 and sum them in
// the same pass. Fusing these was worth doing because at 448 GEMVs per decoded
// token, every extra dependent launch costs ~2-3 us of dead GPU time, and the
// smallest tensors here run in under 3 us.
__global__ void k_prep(float* __restrict__ xf, float* __restrict__ xgsum,
                       const __nv_bfloat16* __restrict__ x, int group) {
  const int g = blockIdx.x;
  const int base = g * group;
  float s = 0.f;
  for (int k = threadIdx.x; k < group; k += blockDim.x) {
    const float v = __bfloat162float(x[base + k]);
    xf[base + k] = v;
    s += v;
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off);
  __shared__ float part[8];
  const int nw = blockDim.x >> 5;
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < nw; ++i) t += part[i];
    xgsum[g] = t;
  }
}

// ---------------------------------------------------------------- gemv
// Lane l owns output row (block_row_base + warp*32 + l). The warp walks the
// input dimension together, so every activation read is a warp-wide broadcast
// and every weight read is 128 contiguous bytes.
template <int WORDS_PER_GROUP>
__global__ __launch_bounds__(BLOCK) void k_gemv(
    float* __restrict__ partial,
    const uint32_t* __restrict__ qw,
    const __nv_bfloat16* __restrict__ sc,
    const uint8_t* __restrict__ zp,
    const float* __restrict__ xf,
    const float* __restrict__ xgsum,
    int out_f, int words_per_row, int G, int groups_per_split, int group) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int o = blockIdx.x * ROWS_PER_BLOCK + warp * WARP + lane;
  if (o >= out_f) return;
  const int b = o >> 5;

  const int g0 = blockIdx.y * groups_per_split;
  const int g1 = min(g0 + groups_per_split, G);

  float acc = 0.f;
  for (int g = g0; g < g1; ++g) {
    const float s = __bfloat162float(sc[(size_t(b) * G + g) * 32 + lane]);
    const float C = 128.0f + float(zp[(size_t(b) * G + g) * 32 + lane]);

    const uint32_t* wp = qw + (size_t(b) * words_per_row + size_t(g) * WORDS_PER_GROUP) * 32 + lane;
    const float*    xp = xf + size_t(g) * group;

    // Eight independent accumulators. One would serialise the whole group into
    // a 128-deep dependent FMA chain (~4 cycles each), which measured as the
    // dominant cost even though the kernel is nominally memory bound.
    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
    float p4 = 0.f, p5 = 0.f, p6 = 0.f, p7 = 0.f;
    #pragma unroll
    for (int t = 0; t < WORDS_PER_GROUP; ++t) {
      const uint32_t w = wp[size_t(t) * 32];
      // All lanes read the same 8 activations: a broadcast, not a gather.
      const float4 xa = *reinterpret_cast<const float4*>(xp + t * 8);
      const float4 xb = *reinterpret_cast<const float4*>(xp + t * 8 + 4);
      const float2 f0 = nib2(w, 0);    // elements 0 and 4
      const float2 f1 = nib2(w, 4);    // elements 1 and 5
      const float2 f2 = nib2(w, 8);    // elements 2 and 6
      const float2 f3 = nib2(w, 12);   // elements 3 and 7
      p0 = fmaf(f0.x, xa.x, p0);  p1 = fmaf(f1.x, xa.y, p1);
      p2 = fmaf(f2.x, xa.z, p2);  p3 = fmaf(f3.x, xa.w, p3);
      p4 = fmaf(f0.y, xb.x, p4);  p5 = fmaf(f1.y, xb.y, p5);
      p6 = fmaf(f2.y, xb.z, p6);  p7 = fmaf(f3.y, xb.w, p7);
    }
    const float pg = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7));
    acc = fmaf(s, pg - C * xgsum[g], acc);
  }
  partial[size_t(blockIdx.y) * out_f + o] = acc;
}

__global__ void k_reduce(float* __restrict__ y, const float* __restrict__ partial,
                         int out_f, int splits) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out_f) return;
  float s = 0.f;
  // Fixed order, so greedy decode stays bitwise reproducible run to run.
  for (int k = 0; k < splits; ++k) s += partial[size_t(k) * out_f + o];
  y[o] = s;
}

__global__ void k_reduce_bf16(__nv_bfloat16* __restrict__ y, const float* __restrict__ partial,
                              int out_f, int splits) {
  const int o = blockIdx.x * blockDim.x + threadIdx.x;
  if (o >= out_f) return;
  float s = 0.f;
  for (int k = 0; k < splits; ++k) s += partial[size_t(k) * out_f + o];
  y[o] = __float2bfloat16(s);
}

int sm_count() {
  static int n = 0;
  if (!n) { cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0); }
  return n;
}

}  // namespace

int gemv_choose_splits(int out_f, int G) {
  // At 4 warps per block, out/32 warps exist before splitting. For out=17408
  // that is 544 warps over 82 SMs -- under 7 warps per SM, far too few to hide
  // DRAM latency. Split the input dimension until the machine is full. The
  // partials this costs are splits*out*4 B, a rounding error against the weight
  // stream (557 KB against 44.6 MB at out=17408).
  const int bx = (out_f + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  static int mult = 0;
  if (!mult) { const char* e = getenv("QWEN_GEMV_WAVES"); mult = e ? atoi(e) : 8; }
  const int target = sm_count() * mult;
  int best = 1;
  for (int s = 1; s <= G; ++s) {
    if (G % s) continue;
    best = s;
    if (bx * s >= target) break;
  }
  return best;
}

void awq_alloc_fused(W4A16Weights& dst, int total_out_f, int in_f, int group,
                     cudaStream_t st) {
  dst.out_f = total_out_f; dst.in_f = in_f; dst.group_size = group;
  dst.num_groups = in_f / group;
  cudaMallocAsync(&dst.qweight, dst.qweight_bytes(), st);
  cudaMallocAsync(&dst.scale,   dst.scale_bytes(),   st);
  cudaMallocAsync(&dst.zp,      dst.zp_bytes(),      st);
}

void awq_repack_into(W4A16Weights& dst, int row_offset, const uint32_t* sq,
                     const __nv_bfloat16* ss, const uint32_t* sz, int out_f,
                     cudaStream_t st) {
  dim3 blk(32, 8);
  k_repack<<<(out_f + 7) / 8, blk, 0, st>>>(dst.qweight, dst.scale, dst.zp, sq, ss, sz,
                                            out_f, dst.in_f / 8, dst.num_groups, row_offset);
}

void awq_repack(W4A16Weights& dst, const uint32_t* sq, const __nv_bfloat16* ss,
                const uint32_t* sz, int out_f, int in_f, int group, cudaStream_t st) {
  awq_alloc_fused(dst, out_f, in_f, group, st);
  awq_repack_into(dst, 0, sq, ss, sz, out_f, st);
}

void awq_free(W4A16Weights& w) {
  cudaFree(w.qweight); cudaFree(w.scale); cudaFree(w.zp);
  w = W4A16Weights{};
}

void gemv_scratch_alloc(GemvScratch& s, int max_in, int max_out, int min_group) {
  s.max_in = max_in; s.max_out = max_out;
  s.max_groups = max_in / min_group;
  s.max_splits = sm_count() * 16;
  cudaMalloc(&s.xf, size_t(max_in) * 4);
  cudaMalloc(&s.xgsum, size_t(s.max_groups) * 4);
  cudaMalloc(&s.partial, size_t(s.max_splits) * max_out * 4);
}

void gemv_scratch_free(GemvScratch& s) {
  cudaFree(s.xf); cudaFree(s.xgsum); cudaFree(s.partial);
  s = GemvScratch{};
}

namespace {
void run(const W4A16Weights& w, const __nv_bfloat16* x, GemvScratch& s,
         int& splits, cudaStream_t st, float* direct) {
  const int G = w.num_groups, wpr = w.in_f / 8;
  splits = gemv_choose_splits(w.out_f, G);
  const int gps = G / splits;

  k_prep<<<G, min(w.group_size, 256), 0, st>>>(s.xf, s.xgsum, x, w.group_size);

  dim3 grid((w.out_f + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, splits);
  float* dst = (splits == 1 && direct) ? direct : s.partial;
  const int wpg = w.group_size / 8;
  if (wpg == 16)
    k_gemv<16><<<grid, BLOCK, 0, st>>>(dst, w.qweight, w.scale, w.zp, s.xf,
                                       s.xgsum, w.out_f, wpr, G, gps, w.group_size);
  else if (wpg == 4)
    k_gemv<4><<<grid, BLOCK, 0, st>>>(dst, w.qweight, w.scale, w.zp, s.xf,
                                      s.xgsum, w.out_f, wpr, G, gps, w.group_size);
  else { fprintf(stderr, "gemv: unsupported group_size %d\n", w.group_size); abort(); }
}
}  // namespace

void gemv_w4a16_f32(float* y, const W4A16Weights& w, const __nv_bfloat16* x,
                    GemvScratch& s, cudaStream_t st) {
  int splits = 1;
  run(w, x, s, splits, st, y);          // splits==1 writes straight to y
  if (splits > 1)
    k_reduce<<<(w.out_f + 255) / 256, 256, 0, st>>>(y, s.partial, w.out_f, splits);
}

void gemv_w4a16(__nv_bfloat16* y, const W4A16Weights& w, const __nv_bfloat16* x,
                GemvScratch& s, cudaStream_t st) {
  int splits = 1;
  run(w, x, s, splits, st, nullptr);
  k_reduce_bf16<<<(w.out_f + 255) / 256, 256, 0, st>>>(y, s.partial, w.out_f, splits);
}

}  // namespace qwen

// ================================================================ W8A16
namespace qwen {
namespace {

// Same lane-owns-a-row mapping as the INT4 path, so activations broadcast and
// weights stay coalesced; only the unpack differs (one byte, symmetric).
__global__ void k_quant_w8(int8_t* __restrict__ q, __nv_bfloat16* __restrict__ sc,
                           const __nv_bfloat16* __restrict__ src,
                           int in_f, int G, int group) {
  const int o = blockIdx.x;
  const int b = o >> 5, l = o & 31;
  for (int g = threadIdx.x; g < G; g += blockDim.x) {
    const size_t base = size_t(o) * in_f + size_t(g) * group;
    float mx = 0.f;
    for (int i = 0; i < group; ++i)
      mx = fmaxf(mx, fabsf(__bfloat162float(src[base + i])));
    const float s = mx / 127.0f;
    const float inv = (mx > 0.f) ? 127.0f / mx : 0.f;
    sc[(size_t(b) * G + g) * 32 + l] = __float2bfloat16(s);
    for (int i = 0; i < group; ++i) {
      const int v = __float2int_rn(__bfloat162float(src[base + i]) * inv);
      q[(size_t(b) * in_f + size_t(g) * group + i) * 32 + l] = int8_t(max(-127, min(127, v)));
    }
  }
}

template <int GROUP>
__global__ __launch_bounds__(BLOCK) void k_gemv8(
    float* __restrict__ partial, const int8_t* __restrict__ qw,
    const __nv_bfloat16* __restrict__ sc, const float* __restrict__ xf,
    int out_f, int in_f, int G, int groups_per_split) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int o = blockIdx.x * ROWS_PER_BLOCK + warp * WARP + lane;
  if (o >= out_f) return;
  const int b = o >> 5;
  const int g0 = blockIdx.y * groups_per_split;
  const int g1 = min(g0 + groups_per_split, G);

  float acc = 0.f;
  for (int g = g0; g < g1; ++g) {
    const float s = __bfloat162float(sc[(size_t(b) * G + g) * 32 + lane]);
    const int8_t* wp = qw + (size_t(b) * in_f + size_t(g) * GROUP) * 32 + lane;
    const float* xp = xf + size_t(g) * GROUP;
    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
    #pragma unroll 8
    for (int t = 0; t < GROUP; t += 4) {
      p0 = fmaf(float(wp[size_t(t) * 32]),       xp[t],     p0);
      p1 = fmaf(float(wp[size_t(t + 1) * 32]),   xp[t + 1], p1);
      p2 = fmaf(float(wp[size_t(t + 2) * 32]),   xp[t + 2], p2);
      p3 = fmaf(float(wp[size_t(t + 3) * 32]),   xp[t + 3], p3);
    }
    acc = fmaf(s, (p0 + p1) + (p2 + p3), acc);
  }
  partial[size_t(blockIdx.y) * out_f + o] = acc;
}

}  // namespace

void quantize_w8a16(W8A16Weights& dst, const __nv_bfloat16* src, int out_f, int in_f,
                    int group, cudaStream_t st) {
  dst.out_f = out_f; dst.in_f = in_f; dst.group_size = group;
  dst.num_groups = in_f / group;
  cudaMalloc(&dst.qweight, size_t(out_f) * in_f);
  cudaMalloc(&dst.scale, size_t(out_f) * dst.num_groups * 2);
  k_quant_w8<<<out_f, 128, 0, st>>>(dst.qweight, dst.scale, src, in_f, dst.num_groups, group);
}

void w8a16_free(W8A16Weights& w) {
  cudaFree(w.qweight); cudaFree(w.scale); w = W8A16Weights{};
}

void gemv_w8a16(__nv_bfloat16* y, const W8A16Weights& w, const __nv_bfloat16* x,
                GemvScratch& s, cudaStream_t st) {
  const int G = w.num_groups;
  const int splits = gemv_choose_splits(w.out_f, G);
  const int gps = G / splits;
  k_prep<<<G, min(w.group_size, 256), 0, st>>>(s.xf, s.xgsum, x, w.group_size);
  dim3 grid((w.out_f + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, splits);
  if (w.group_size == 128)
    k_gemv8<128><<<grid, BLOCK, 0, st>>>(s.partial, w.qweight, w.scale, s.xf,
                                         w.out_f, w.in_f, G, gps);
  else { fprintf(stderr, "gemv_w8a16: unsupported group %d\n", w.group_size); abort(); }
  k_reduce_bf16<<<(w.out_f + 255) / 256, 256, 0, st>>>(y, s.partial, w.out_f, splits);
}

}  // namespace qwen
