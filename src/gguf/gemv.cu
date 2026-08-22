#include "gemv.cuh"
#include "ggml_tables.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>

namespace qwen {
namespace {

#define QK_K 256

__device__ __forceinline__ float h2f(uint16_t h) {
  return __half2float(__ushort_as_half(h));
}
__device__ __forceinline__ uint32_t le32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

// Same layouts as src/gguf/dequant.cu, which is gated bit-exact against ggml.
struct BQ2K { uint8_t scales[16]; uint8_t qs[64]; uint16_t d, dmin; };
struct BQ3K { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; };
struct BQ4K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; };
struct BQ5K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct BQ6K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
struct BQ80 { uint16_t d; int8_t qs[32]; };
struct BIQ4NL { uint16_t d; uint8_t qs[16]; };
struct BIQ4XS { uint16_t d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };
struct BIQ2XS { uint16_t d; uint16_t qs[32]; uint8_t scales[8]; };
struct BIQ2S { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t scales[8]; };
struct BIQ2XXS { uint16_t d; uint16_t qs[32]; };
struct BIQ3XXS { uint16_t d; uint8_t qs[96]; };
struct BIQ3S { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32]; uint8_t scales[4]; };

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
  else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
  }
}

struct Tab {
  const int8_t* kv4nl;
  const uint64_t* iq2xxs;
  const uint64_t* iq2xs;
  const uint64_t* iq2s;
  const uint32_t* iq3xxs;
  const uint32_t* iq3s;
  const uint8_t* ksigns;
  const uint8_t* kmask;
};

// ---------------------------------------------------------------- deq8
//
// Produce the 8 values at [o0, o0+8) of a 256-element run. o0 is always a
// multiple of 8, which is what keeps every lane inside one sub-block and one
// scale.
template <GgmlType T> struct Deq;

template <> struct Deq<GgmlType::Q4_K> {
  static constexpr int RUN_BYTES = 144;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab&) {
    const BQ4K* b = reinterpret_cast<const BQ4K*>(p);
    const float d = h2f(b->d), dm = h2f(b->dmin);
    const int g = o0 / 64, within = o0 % 64, half = within / 32, l = within % 32;
    uint8_t sc, m;
    scale_min_k4(2 * g + half, b->scales, sc, m);
    const float dl = d * sc, ml = dm * m;
    // One 64-bit load instead of eight 1-byte loads. This is safe without any
    // repack because a Q4_K block is 144 bytes (a multiple of 16), qs sits at
    // offset 16, and l is always a multiple of 8 -- see the alignment contract
    // on gguf_gemv().
    const uint2 qv = *reinterpret_cast<const uint2*>(b->qs + g * 32 + l);
    const int lsh = half ? 4 : 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t wq = (i < 4) ? qv.x : qv.y;
      y[i] = dl * float((wq >> (8 * (i & 3) + lsh)) & 0xF) - ml;
    }
  }
};

template <> struct Deq<GgmlType::Q5_K> {
  static constexpr int RUN_BYTES = 176;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab&) {
    const BQ5K* b = reinterpret_cast<const BQ5K*>(p);
    const float d = h2f(b->d), dm = h2f(b->dmin);
    const int g = o0 / 64, within = o0 % 64, half = within / 32, l = within % 32;
    uint8_t sc, m;
    scale_min_k4(2 * g + half, b->scales, sc, m);
    const float dl = d * sc, ml = dm * m;
    // Same 64-bit trick as Q4_K, twice: a Q5_K block is 176 bytes with qs at
    // offset 48 and qh at 16, both 8-aligned, and l is a multiple of 8. This is
    // the tensor that matters most in Q3_K_XL -- output.weight is 834 MiB of
    // Q5_K, more than every other tensor in the sweep put together.
    const uint2 qv = *reinterpret_cast<const uint2*>(b->qs + g * 32 + l);
    const uint2 hv = *reinterpret_cast<const uint2*>(b->qh + l);
    const int lsh = half ? 4 : 0;
    const uint32_t bit = 1u << (2 * g + half);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t wq = (i < 4) ? qv.x : qv.y;
      const uint32_t wh = (i < 4) ? hv.x : hv.y;
      const int sh = 8 * (i & 3);
      const float lo = float((wq >> (sh + lsh)) & 0xF);
      y[i] = dl * (lo + (((wh >> sh) & bit) ? 16.f : 0.f)) - ml;
    }
  }
};

template <> struct Deq<GgmlType::Q6_K> {
  static constexpr int RUN_BYTES = 210;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab&) {
    const BQ6K* b = reinterpret_cast<const BQ6K*>(p);
    const float d = h2f(b->d);
    const int n = o0 / 128, r = o0 % 128, sub = r / 32, l = r % 32;
    const uint8_t* ql = b->ql + n * 64;
    const uint8_t* qh = b->qh + n * 32;
    const int8_t* sc = b->scales + n * 8;
    const int is = l / 16;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int li = l + i;
      const uint8_t h = qh[li];
      int q, s;
      if (sub == 0)      { q = int(ql[li]      & 0xF) | int(((h >> 0) & 3) << 4); s = sc[is + 0]; }
      else if (sub == 1) { q = int(ql[li + 32] & 0xF) | int(((h >> 2) & 3) << 4); s = sc[is + 2]; }
      else if (sub == 2) { q = int(ql[li]      >> 4)  | int(((h >> 4) & 3) << 4); s = sc[is + 4]; }
      else               { q = int(ql[li + 32] >> 4)  | int(((h >> 6) & 3) << 4); s = sc[is + 6]; }
      y[i] = d * float(s) * float(q - 32);
    }
  }
};

template <> struct Deq<GgmlType::Q3_K> {
  static constexpr int RUN_BYTES = 110;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab&) {
    const BQ3K* b = reinterpret_cast<const BQ3K*>(p);
    const int n = o0 / 128, r = o0 % 128, j = r / 32, within = r % 32;
    const int half = within / 16, l = within % 16;
    const int is = 2 * (4 * n + j) + half;

    // Only ONE of the 16 six-bit scales is needed here, so extract just that
    // byte. ggml's reference unpacks all sixteen into a 4-word aux[] shuffle,
    // and this kernel calls get8 once per lane per super-block -- so every lane
    // in the warp was redoing the full unpack (~20 ALU ops) to read one byte of
    // it. Decode is ALU bound in this kernel, not bandwidth bound (stubbing the
    // dequant out measures 646 GB/s against 347 with it), so this is on the
    // critical path.
    //
    // The layout, from the reference shuffle: with k = is & 3 and w = is >> 2,
    // the low nibble comes from scales[k] or scales[4+k] (w odd picks the
    // second group, w >= 2 picks the high nibble) and the top two bits come
    // from scales[8+k] shifted by 2*w. Verified bit-exact against the
    // full-block dequantiser by gate_gguf_gemv.
    const int k = is & 3, w = is >> 2;
    const uint8_t lo_src = b->scales[(w & 1) ? 4 + k : k];
    const uint8_t lo = (w & 2) ? uint8_t(lo_src >> 4) : uint8_t(lo_src & 0xF);
    const uint8_t sc6 = uint8_t(lo | (((b->scales[8 + k] >> (2 * w)) & 3) << 4));
    const float dl = h2f(b->d) * float(int(sc6) - 32);
    const uint8_t m = uint8_t(1u << (4 * n + j));
    const int shift = 2 * j;
    const uint8_t* q = b->qs + n * 32 + 16 * half + l;
    const uint8_t* hm = b->hmask + 16 * half + l;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      y[i] = dl * (float(int8_t((q[i] >> shift) & 3)) - ((hm[i] & m) ? 0.f : 4.f));
  }
};

template <> struct Deq<GgmlType::Q2_K> {
  static constexpr int RUN_BYTES = 84;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab&) {
    const BQ2K* b = reinterpret_cast<const BQ2K*>(p);
    const float d = h2f(b->d), dm = h2f(b->dmin);
    const int n = o0 / 128, r = o0 % 128, j = r / 32, within = r % 32;
    const int half = within / 16, l = within % 16;
    const uint8_t sc = b->scales[2 * (4 * n + j) + half];
    const float dl = d * float(sc & 0xF), ml = dm * float(sc >> 4);
    const int shift = 2 * j;
    const uint8_t* q = b->qs + n * 32 + 16 * half + l;
    #pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = dl * float((q[i] >> shift) & 3) - ml;
  }
};

template <> struct Deq<GgmlType::IQ4_XS> {
  static constexpr int RUN_BYTES = 136;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab& t) {
    const BIQ4XS* b = reinterpret_cast<const BIQ4XS*>(p);
    const int ib = o0 / 32, within = o0 % 32;
    const int ls = ((b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                   (((b->scales_h >> (2 * ib)) & 3) << 4);
    const float dl = h2f(b->d) * float(ls - 32);
    // One 64-bit load rather than eight 1-byte loads: an IQ4_XS block is 136
    // bytes (8 x 17) with qs at offset 8, and `within % 16` is 0 or 8, so the
    // address is 8-aligned in every block. IQ4_XS is 38% of UD-Q3_K_XL by
    // bytes -- the largest single share in the file.
    const uint2 qv = *reinterpret_cast<const uint2*>(b->qs + ib * 16 + (within % 16));
    const int lsh = (within >= 16) ? 4 : 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t wq = (i < 4) ? qv.x : qv.y;
      y[i] = dl * float(t.kv4nl[(wq >> (8 * (i & 3) + lsh)) & 0xF]);
    }
  }
};

template <> struct Deq<GgmlType::IQ2_XS> {
  static constexpr int RUN_BYTES = 74;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab& t) {
    const BIQ2XS* b = reinterpret_cast<const BIQ2XS*>(p);
    const int ib32 = o0 / 32, l = (o0 % 32) / 8;
    const uint16_t qi = b->qs[4 * ib32 + l];
    const float db = h2f(b->d) *
        (0.5f + float((b->scales[ib32] >> (4 * (l / 2))) & 0xF)) * 0.25f;
    const uint64_t g = t.iq2xs[qi & 511];
    const uint8_t* gp = reinterpret_cast<const uint8_t*>(&g);
    const uint8_t sg = t.ksigns[qi >> 9];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      y[i] = db * float(gp[i]) * ((sg & (1u << i)) ? -1.f : 1.f);
  }
};

template <> struct Deq<GgmlType::IQ2_XXS> {
  static constexpr int RUN_BYTES = 66;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab& t) {
    const BIQ2XXS* b = reinterpret_cast<const BIQ2XXS*>(p);
    const int ib32 = o0 / 32, l = (o0 % 32) / 8;
    const uint16_t* q16 = b->qs + 4 * ib32;
    const uint32_t a0 = uint32_t(q16[0]) | (uint32_t(q16[1]) << 16);
    const uint32_t a1 = uint32_t(q16[2]) | (uint32_t(q16[3]) << 16);
    const uint8_t* a8 = reinterpret_cast<const uint8_t*>(&a0);
    const float db = h2f(b->d) * (0.5f + float(a1 >> 28)) * 0.25f;
    const uint64_t g = t.iq2xxs[a8[l]];
    const uint8_t* gp = reinterpret_cast<const uint8_t*>(&g);
    const uint8_t sg = t.ksigns[(a1 >> (7 * l)) & 127];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      y[i] = db * float(gp[i]) * ((sg & (1u << i)) ? -1.f : 1.f);
  }
};

template <> struct Deq<GgmlType::IQ2_S> {
  static constexpr int RUN_BYTES = 82;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab& t) {
    const BIQ2S* b = reinterpret_cast<const BIQ2S*>(p);
    const int ib32 = o0 / 32, l = (o0 % 32) / 8;
    const float db = h2f(b->d) *
        (0.5f + float((b->scales[ib32] >> (4 * (l / 2))) & 0xF)) * 0.25f;
    const uint64_t g = t.iq2s[b->qs[4 * ib32 + l] |
                              ((uint32_t(b->qh[ib32]) << (8 - 2 * l)) & 0x300)];
    const uint8_t* gp = reinterpret_cast<const uint8_t*>(&g);
    const uint8_t sg = b->qs[32 + 4 * ib32 + l];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      y[i] = db * float(gp[i]) * ((sg & (1u << i)) ? -1.f : 1.f);
  }
};

template <> struct Deq<GgmlType::IQ3_XXS> {
  static constexpr int RUN_BYTES = 98;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab& t) {
    const BIQ3XXS* b = reinterpret_cast<const BIQ3XXS*>(p);
    const int ib32 = o0 / 32, l = (o0 % 32) / 8;
    const uint32_t aux32 = le32(b->qs + 64 + 4 * ib32);
    const float db = h2f(b->d) * (0.5f + float(aux32 >> 28)) * 0.5f;
    const uint8_t sg = t.ksigns[(aux32 >> (7 * l)) & 127];
    const uint32_t g1 = t.iq3xxs[b->qs[8 * ib32 + 2 * l + 0]];
    const uint32_t g2 = t.iq3xxs[b->qs[8 * ib32 + 2 * l + 1]];
    const uint8_t* p1 = reinterpret_cast<const uint8_t*>(&g1);
    const uint8_t* p2 = reinterpret_cast<const uint8_t*>(&g2);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      y[i]     = db * float(p1[i]) * ((sg & (1u << i))     ? -1.f : 1.f);
      y[i + 4] = db * float(p2[i]) * ((sg & (1u << (i + 4))) ? -1.f : 1.f);
    }
  }
};

template <> struct Deq<GgmlType::IQ3_S> {
  static constexpr int RUN_BYTES = 110;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab& t) {
    const BIQ3S* b = reinterpret_cast<const BIQ3S*>(p);
    const int s = o0 / 32, l = (o0 % 32) / 8;
    const float db = h2f(b->d) *
        (1.f + 2.f * float((b->scales[s / 2] >> (4 * (s % 2))) & 0xF));
    const uint8_t qh = b->qh[s];
    const uint32_t g1 = t.iq3s[b->qs[8 * s + 2 * l + 0] |
                               ((uint32_t(qh) << (8 - 2 * l)) & 256)];
    const uint32_t g2 = t.iq3s[b->qs[8 * s + 2 * l + 1] |
                               ((uint32_t(qh) << (7 - 2 * l)) & 256)];
    const uint8_t* p1 = reinterpret_cast<const uint8_t*>(&g1);
    const uint8_t* p2 = reinterpret_cast<const uint8_t*>(&g2);
    const uint8_t sg = b->signs[4 * s + l];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      y[i]     = db * float(p1[i]) * ((sg & (1u << i))     ? -1.f : 1.f);
      y[i + 4] = db * float(p2[i]) * ((sg & (1u << (i + 4))) ? -1.f : 1.f);
    }
  }
};

// 32-element blocks: a 256-run is 8 of them, and a lane's 8 values sit inside
// block o0/32.
template <> struct Deq<GgmlType::Q8_0> {
  static constexpr int RUN_BYTES = 8 * 34;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab&) {
    const BQ80* b = reinterpret_cast<const BQ80*>(p) + (o0 / 32);
    const float d = h2f(b->d);
    const int w = o0 % 32;
    #pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = d * float(b->qs[w + i]);
  }
};

template <> struct Deq<GgmlType::IQ4_NL> {
  static constexpr int RUN_BYTES = 8 * 18;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab& t) {
    const BIQ4NL* b = reinterpret_cast<const BIQ4NL*>(p) + (o0 / 32);
    const float d = h2f(b->d);
    const int w = o0 % 32;
    const uint8_t* q = b->qs + (w % 16);
    const bool hi = w >= 16;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      y[i] = d * float(t.kv4nl[hi ? (q[i] >> 4) : (q[i] & 0xF)]);
  }
};

template <> struct Deq<GgmlType::F32> {
  static constexpr int RUN_BYTES = 256 * 4;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab&) {
    const float* f = reinterpret_cast<const float*>(p) + o0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = f[i];
  }
};
template <> struct Deq<GgmlType::F16> {
  static constexpr int RUN_BYTES = 256 * 2;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab&) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p) + o0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) y[i] = h2f(h[i]);
  }
};
template <> struct Deq<GgmlType::BF16> {
  static constexpr int RUN_BYTES = 256 * 2;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab&) {
    const uint16_t* h = reinterpret_cast<const uint16_t*>(p) + o0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t u = uint32_t(h[i]) << 16;
      float f; memcpy(&f, &u, 4);
      y[i] = f;
    }
  }
};

// ---------------------------------------------------------------- kernels
constexpr int GEMV_WARPS = 4;
constexpr int GEMV_THREADS = GEMV_WARPS * 32;

template <GgmlType T, int MROWS>
__global__ __launch_bounds__(GEMV_THREADS) void k_gemv(
    __nv_bfloat16* __restrict__ y, const uint8_t* __restrict__ w,
    const __nv_bfloat16* __restrict__ x, int out_f, int in_f, int ldy, Tab tab) {
  const int warp = (blockIdx.x * GEMV_WARPS) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= out_f) return;
  const int runs = in_f >> 8;
  const uint8_t* wr = w + size_t(warp) * size_t(runs) * Deq<T>::RUN_BYTES;
  const int o0 = lane * 8;

  float acc[MROWS];
  #pragma unroll
  for (int m = 0; m < MROWS; ++m) acc[m] = 0.f;

  for (int i = 0; i < runs; ++i) {
    float v[8];
    Deq<T>::get8(wr + size_t(i) * Deq<T>::RUN_BYTES, o0, v, tab);
    const int base = (i << 8) + o0;
    #pragma unroll
    for (int m = 0; m < MROWS; ++m) {
      const __nv_bfloat16* xp = x + size_t(m) * in_f + base;
      #pragma unroll
      for (int j = 0; j < 8; ++j) acc[m] = fmaf(v[j], __bfloat162float(xp[j]), acc[m]);
    }
  }
  #pragma unroll
  for (int m = 0; m < MROWS; ++m) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
    if (lane == 0) y[size_t(m) * ldy + warp] = __float2bfloat16(acc[m]);
  }
}

#define CKG(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); abort(); } } while(0)

template <class T> T* up(const T* h, size_t n) {
  T* d = nullptr;
  CKG(cudaMalloc(&d, n * sizeof(T)));
  CKG(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

Tab g_tab;
bool g_ready = false;
Tab tables() {
  if (!g_ready) {
    g_tab.kv4nl  = up(kvalues_iq4nl, 16);
    g_tab.iq2xxs = up(iq2xxs_grid, 256);
    g_tab.iq2xs  = up(iq2xs_grid, 512);
    g_tab.iq2s   = up(iq2s_grid, 1024);
    g_tab.iq3xxs = up(iq3xxs_grid, 256);
    g_tab.iq3s   = up(iq3s_grid, 512);
    g_tab.ksigns = up(ksigns_iq2xs, 128);
    g_tab.kmask  = up(kmask_iq2xs, 8);
    g_ready = true;
  }
  return g_tab;
}

// One thread per 8 values, walking runs of 256 exactly as the GEMV does.
template <GgmlType T>
__global__ void k_deq8_dump(float* __restrict__ dst, const uint8_t* __restrict__ src,
                            int64_t n8, Tab tab) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n8) return;
  const int64_t run = i >> 5;             // 32 groups of 8 per 256-run
  const int o0 = int(i & 31) * 8;
  float v[8];
  Deq<T>::get8(src + run * Deq<T>::RUN_BYTES, o0, v, tab);
  #pragma unroll
  for (int j = 0; j < 8; ++j) dst[i * 8 + j] = v[j];
}

template <GgmlType T>
void launch_dump(float* dst, const void* src, int64_t n, cudaStream_t st) {
  const int64_t n8 = n / 8;
  const Tab tb = tables();
  k_deq8_dump<T><<<int((n8 + 255) / 256), 256, 0, st>>>(
      dst, static_cast<const uint8_t*>(src), n8, tb);
}

template <GgmlType T>
void launch(__nv_bfloat16* y, const GgufWeight& w, const __nv_bfloat16* x, int M,
            int ldy, cudaStream_t st) {
  const int blocks = (w.out_f + GEMV_WARPS - 1) / GEMV_WARPS;
  const Tab tb = tables();
  const uint8_t* p = static_cast<const uint8_t*>(w.data);
  if (ldy <= 0) ldy = w.out_f;
  // Every M from 1 to 8, not just the powers of two. The weight stream is the
  // whole cost here, so a caller with M=7 that had to split into 4+2+1 read the
  // tensor THREE times. That is exactly what speculative decoding asks for --
  // the drafter proposes block_size-1 = 7 rows through the lm_head every round,
  // and on a 0.814 GiB GGUF head the split cost about 2.4 GiB of reads per
  // round instead of 0.8.
  switch (M) {
#define QWEN_M_CASE(N) \
    case N: k_gemv<T, N><<<blocks, GEMV_THREADS, 0, st>>>(y, p, x, w.out_f, w.in_f, ldy, tb); break;
    QWEN_M_CASE(1) QWEN_M_CASE(2) QWEN_M_CASE(3) QWEN_M_CASE(4)
    QWEN_M_CASE(5) QWEN_M_CASE(6) QWEN_M_CASE(7) QWEN_M_CASE(8)
#undef QWEN_M_CASE
    default:
      fprintf(stderr, "gguf gemv: unsupported M %d\n", M);
      abort();
  }
}

#define DISPATCH(FN, ...)                                              \
  switch (w.type) {                                                    \
    case GgmlType::Q2_K:    FN<GgmlType::Q2_K>(__VA_ARGS__); break;     \
    case GgmlType::Q3_K:    FN<GgmlType::Q3_K>(__VA_ARGS__); break;     \
    case GgmlType::Q4_K:    FN<GgmlType::Q4_K>(__VA_ARGS__); break;     \
    case GgmlType::Q5_K:    FN<GgmlType::Q5_K>(__VA_ARGS__); break;     \
    case GgmlType::Q6_K:    FN<GgmlType::Q6_K>(__VA_ARGS__); break;     \
    case GgmlType::Q8_0:    FN<GgmlType::Q8_0>(__VA_ARGS__); break;     \
    case GgmlType::IQ2_XS:  FN<GgmlType::IQ2_XS>(__VA_ARGS__); break;   \
    case GgmlType::IQ2_XXS: FN<GgmlType::IQ2_XXS>(__VA_ARGS__); break;  \
    case GgmlType::IQ2_S:   FN<GgmlType::IQ2_S>(__VA_ARGS__); break;    \
    case GgmlType::IQ3_XXS: FN<GgmlType::IQ3_XXS>(__VA_ARGS__); break;  \
    case GgmlType::IQ3_S:   FN<GgmlType::IQ3_S>(__VA_ARGS__); break;    \
    case GgmlType::IQ4_NL:  FN<GgmlType::IQ4_NL>(__VA_ARGS__); break;   \
    case GgmlType::IQ4_XS:  FN<GgmlType::IQ4_XS>(__VA_ARGS__); break;   \
    case GgmlType::F32:     FN<GgmlType::F32>(__VA_ARGS__); break;      \
    case GgmlType::F16:     FN<GgmlType::F16>(__VA_ARGS__); break;      \
    case GgmlType::BF16:    FN<GgmlType::BF16>(__VA_ARGS__); break;     \
    default:                                                            \
      fprintf(stderr, "gguf gemv: type %s not implemented\n",           \
              ggml_type_name(w.type));                                  \
      abort();                                                          \
  }

}  // namespace

bool gguf_gemv_supported(GgmlType t) {
  switch (t) {
    case GgmlType::Q2_K: case GgmlType::Q3_K: case GgmlType::Q4_K:
    case GgmlType::Q5_K: case GgmlType::Q6_K: case GgmlType::Q8_0:
    case GgmlType::IQ2_XS: case GgmlType::IQ2_XXS: case GgmlType::IQ2_S:
    case GgmlType::IQ3_XXS:
    case GgmlType::IQ3_S: case GgmlType::IQ4_NL: case GgmlType::IQ4_XS:
    case GgmlType::F32: case GgmlType::F16: case GgmlType::BF16:
      return true;
    default: return false;
  }
}

size_t gguf_row_bytes(GgmlType t, int in_f) {
  return size_t(in_f) / size_t(ggml_block_elems(t)) * size_t(ggml_block_bytes(t));
}

void gguf_deq8_dump(float* dst, const void* src, GgmlType type, int64_t n,
                    cudaStream_t st) {
  if (n % 256) { fprintf(stderr, "deq8 dump: n must be a multiple of 256\n"); abort(); }
  GgufWeight w; w.type = type;   // DISPATCH reads w.type
  DISPATCH(launch_dump, dst, src, n, st);
}

// The Q4_K and Q5_K paths issue 64-bit loads into the block, which is only
// legal if the tensor base is 16-byte aligned (see gemv.cuh). A misaligned
// load is an illegal memory access reported at some later, unrelated kernel, so
// check it here where the message can name the cause.
static void check_align(const GgufWeight& w, const char* who) {
  if (reinterpret_cast<uintptr_t>(w.data) % 16) {
    fprintf(stderr, "%s: weight data %p is not 16-byte aligned\n", who, w.data);
    abort();
  }
}

void gguf_gemv(__nv_bfloat16* y, const GgufWeight& w, const __nv_bfloat16* x,
               cudaStream_t st, int ldy) {
  if (w.in_f % 256) { fprintf(stderr, "gguf gemv: in_f %d is not a multiple of 256\n",
                              w.in_f); abort(); }
  check_align(w, "gguf_gemv");
  DISPATCH(launch, y, w, x, 1, ldy, st);
}

void gguf_gemm_small(__nv_bfloat16* y, const GgufWeight& w, const __nv_bfloat16* x,
                     int M, cudaStream_t st, int ldy) {
  if (w.in_f % 256) { fprintf(stderr, "gguf gemm: in_f %d is not a multiple of 256\n",
                              w.in_f); abort(); }
  check_align(w, "gguf_gemm_small");
  DISPATCH(launch, y, w, x, M, ldy, st);
}

}  // namespace qwen
