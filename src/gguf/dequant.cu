#include "dequant.cuh"
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

// ---- block layouts, byte for byte as ggml writes them -------------------
// Named fields rather than offsets so a mismatch is a compile error and not a
// silent misread. Sizes are asserted against ggml_block_bytes() at launch.
struct BQ2K { uint8_t scales[QK_K / 16]; uint8_t qs[QK_K / 4]; uint16_t d, dmin; };
struct BQ3K { uint8_t hmask[QK_K / 8]; uint8_t qs[QK_K / 4]; uint8_t scales[12]; uint16_t d; };
struct BQ4K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[QK_K / 2]; };
struct BQ5K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[QK_K / 8]; uint8_t qs[QK_K / 2]; };
struct BQ6K { uint8_t ql[QK_K / 2]; uint8_t qh[QK_K / 4]; int8_t scales[QK_K / 16]; uint16_t d; };
struct BQ80 { uint16_t d; int8_t qs[32]; };
struct BIQ4NL { uint16_t d; uint8_t qs[16]; };
struct BIQ4XS { uint16_t d; uint16_t scales_h; uint8_t scales_l[QK_K / 64]; uint8_t qs[QK_K / 2]; };
struct BIQ2XS { uint16_t d; uint16_t qs[QK_K / 8]; uint8_t scales[QK_K / 32]; };
struct BIQ2S { uint16_t d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32];
               uint8_t scales[QK_K / 32]; };
struct BIQ2XXS { uint16_t d; uint16_t qs[QK_K / 8]; };
struct BIQ3XXS { uint16_t d; uint8_t qs[3 * QK_K / 8]; };
struct BIQ3S { uint16_t d; uint8_t qs[QK_K / 4]; uint8_t qh[QK_K / 32];
               uint8_t signs[QK_K / 8]; uint8_t scales[QK_K / 64]; };

// The 6-bit packed scale/min pair the K-quants share.
__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
  else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
  }
}

// Each thread owns one super-block. Parallelism comes from block count, which is
// large for any real tensor (a 17408x5120 weight is 348k super-blocks).
template <class Out>
__device__ __forceinline__ void put(Out* y, int i, float v);
template <> __device__ __forceinline__ void put<float>(float* y, int i, float v) { y[i] = v; }
template <> __device__ __forceinline__ void put<__nv_bfloat16>(__nv_bfloat16* y, int i, float v) {
  y[i] = __float2bfloat16(v);
}

template <class Out>
__global__ void k_q2k(Out* out, const BQ2K* x, int64_t nb) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * QK_K;
  const float d = h2f(x[i].d), mn = h2f(x[i].dmin);
  const uint8_t* q = x[i].qs;
  int is = 0, o = 0;
  for (int n = 0; n < QK_K; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      uint8_t sc = x[i].scales[is++];
      float dl = d * (sc & 0xF), ml = mn * (sc >> 4);
      for (int l = 0; l < 16; ++l) put(y, o++, dl * float((q[l] >> shift) & 3) - ml);
      sc = x[i].scales[is++];
      dl = d * (sc & 0xF); ml = mn * (sc >> 4);
      for (int l = 0; l < 16; ++l) put(y, o++, dl * float((q[l + 16] >> shift) & 3) - ml);
      shift += 2;
    }
    q += 32;
  }
}

template <class Out>
__global__ void k_q3k(Out* out, const BQ3K* x, int64_t nb) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * QK_K;
  const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
  uint32_t aux[4];
  #pragma unroll
  for (int t = 0; t < 3; ++t) {
    uint32_t v = 0;
    for (int b = 0; b < 4; ++b) v |= uint32_t(x[i].scales[4 * t + b]) << (8 * b);
    aux[t] = v;
  }
  const uint32_t tmp = aux[2];
  aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

  const float d_all = h2f(x[i].d);
  const uint8_t* q = x[i].qs;
  const uint8_t* hm = x[i].hmask;
  uint8_t m = 1;
  int is = 0, o = 0;
  for (int n = 0; n < QK_K; n += 128) {
    int shift = 0;
    for (int j = 0; j < 4; ++j) {
      float dl = d_all * float(scales[is++] - 32);
      for (int l = 0; l < 16; ++l)
        put(y, o++, dl * (float(int8_t((q[l] >> shift) & 3)) - ((hm[l] & m) ? 0.f : 4.f)));
      dl = d_all * float(scales[is++] - 32);
      for (int l = 0; l < 16; ++l)
        put(y, o++, dl * (float(int8_t((q[l + 16] >> shift) & 3)) - ((hm[l + 16] & m) ? 0.f : 4.f)));
      shift += 2;
      m <<= 1;
    }
    q += 32;
  }
}

template <class Out>
__global__ void k_q4k(Out* out, const BQ4K* x, int64_t nb) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * QK_K;
  const float d = h2f(x[i].d), mn = h2f(x[i].dmin);
  const uint8_t* q = x[i].qs;
  int is = 0, o = 0;
  for (int j = 0; j < QK_K; j += 64) {
    uint8_t sc, m;
    scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = d * sc, m1 = mn * m;
    scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = d * sc, m2 = mn * m;
    for (int l = 0; l < 32; ++l) put(y, o++, d1 * float(q[l] & 0xF) - m1);
    for (int l = 0; l < 32; ++l) put(y, o++, d2 * float(q[l] >> 4) - m2);
    q += 32; is += 2;
  }
}

template <class Out>
__global__ void k_q5k(Out* out, const BQ5K* x, int64_t nb) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * QK_K;
  const float d = h2f(x[i].d), mn = h2f(x[i].dmin);
  const uint8_t* ql = x[i].qs;
  const uint8_t* qh = x[i].qh;
  int is = 0, o = 0;
  uint8_t u1 = 1, u2 = 2;
  for (int j = 0; j < QK_K; j += 64) {
    uint8_t sc, m;
    scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = d * sc, m1 = mn * m;
    scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = d * sc, m2 = mn * m;
    for (int l = 0; l < 32; ++l)
      put(y, o++, d1 * (float(ql[l] & 0xF) + ((qh[l] & u1) ? 16.f : 0.f)) - m1);
    for (int l = 0; l < 32; ++l)
      put(y, o++, d2 * (float(ql[l] >> 4) + ((qh[l] & u2) ? 16.f : 0.f)) - m2);
    ql += 32; is += 2; u1 <<= 2; u2 <<= 2;
  }
}

template <class Out>
__global__ void k_q6k(Out* out, const BQ6K* x, int64_t nb) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * QK_K;
  const float d = h2f(x[i].d);
  const uint8_t* ql = x[i].ql;
  const uint8_t* qh = x[i].qh;
  const int8_t* sc = x[i].scales;
  int base = 0;
  for (int n = 0; n < QK_K; n += 128) {
    for (int l = 0; l < 32; ++l) {
      const int is = l / 16;
      const int8_t q1 = int8_t((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      const int8_t q2 = int8_t((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      const int8_t q3 = int8_t((ql[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      const int8_t q4 = int8_t((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      put(y, base + l +  0, d * float(sc[is + 0]) * float(q1));
      put(y, base + l + 32, d * float(sc[is + 2]) * float(q2));
      put(y, base + l + 64, d * float(sc[is + 4]) * float(q3));
      put(y, base + l + 96, d * float(sc[is + 6]) * float(q4));
    }
    base += 128; ql += 64; qh += 32; sc += 8;
  }
}

template <class Out>
__global__ void k_q80(Out* out, const BQ80* x, int64_t nb) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * 32;
  const float d = h2f(x[i].d);
  for (int j = 0; j < 32; ++j) put(y, j, d * float(x[i].qs[j]));
}

template <class Out>
__global__ void k_iq4nl(Out* out, const BIQ4NL* x, int64_t nb, const int8_t* kv) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * 32;
  const float d = h2f(x[i].d);
  for (int j = 0; j < 16; ++j) {
    put(y, j,      d * float(kv[x[i].qs[j] & 0xF]));
    put(y, j + 16, d * float(kv[x[i].qs[j] >> 4]));
  }
}

template <class Out>
__global__ void k_iq4xs(Out* out, const BIQ4XS* x, int64_t nb, const int8_t* kv) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * QK_K;
  const float d = h2f(x[i].d);
  const uint8_t* qs = x[i].qs;
  int o = 0;
  for (int ib = 0; ib < QK_K / 32; ++ib) {
    // 6-bit sub-block scale: 4 low bits from scales_l, 2 high bits from scales_h
    const int ls = ((x[i].scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                   (((x[i].scales_h >> (2 * ib)) & 3) << 4);
    const float dl = d * float(ls - 32);
    for (int j = 0; j < 16; ++j) {
      put(y, o + j,      dl * float(kv[qs[j] & 0xF]));
      put(y, o + j + 16, dl * float(kv[qs[j] >> 4]));
    }
    qs += 16; o += 32;
  }
}

template <class Out>
__global__ void k_iq2xs(Out* out, const BIQ2XS* x, int64_t nb,
                        const uint64_t* grid, const uint8_t* ksigns, const uint8_t* kmask) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * QK_K;
  const float d = h2f(x[i].d);
  int o = 0;
  for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
    const float db0 = d * (0.5f + float(x[i].scales[ib32] & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + float(x[i].scales[ib32] >> 4)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
      const uint64_t g = grid[x[i].qs[4 * ib32 + l] & 511];
      const uint8_t* gp = reinterpret_cast<const uint8_t*>(&g);
      const uint8_t signs = ksigns[x[i].qs[4 * ib32 + l] >> 9];
      const float db = (l / 2) ? db1 : db0;
      for (int j = 0; j < 8; ++j)
        put(y, o + j, db * float(gp[j]) * ((signs & kmask[j]) ? -1.f : 1.f));
      o += 8;
    }
  }
}

// IQ2_XXS: the grid index and the sign/scale word are packed into two uint32s
// per 32-element group -- aux32[0]'s four bytes are the grid indices, aux32[1]
// carries four 7-bit sign selectors and a 4-bit scale in its top nibble.
template <class Out>
__global__ void k_iq2xxs(Out* out, const BIQ2XXS* x, int64_t nb,
                         const uint64_t* grid, const uint8_t* ksigns, const uint8_t* kmask) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * QK_K;
  const float d = h2f(x[i].d);
  int o = 0;
  for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
    const uint16_t* q16 = x[i].qs + 4 * ib32;
    uint32_t a0 = uint32_t(q16[0]) | (uint32_t(q16[1]) << 16);
    uint32_t a1 = uint32_t(q16[2]) | (uint32_t(q16[3]) << 16);
    const uint8_t* a8 = reinterpret_cast<const uint8_t*>(&a0);
    const float db = d * (0.5f + float(a1 >> 28)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
      const uint64_t g = grid[a8[l]];
      const uint8_t* gp = reinterpret_cast<const uint8_t*>(&g);
      const uint8_t sg = ksigns[(a1 >> (7 * l)) & 127];
      for (int j = 0; j < 8; ++j)
        put(y, o + j, db * float(gp[j]) * ((sg & kmask[j]) ? -1.f : 1.f));
      o += 8;
    }
  }
}

// IQ2_S: like IQ2_XS but the grid index gets 2 extra bits from qh, and the sign
// bytes live INSIDE qs (at qs + QK_K/8) rather than being packed into the index.
template <class Out>
__global__ void k_iq2s(Out* out, const BIQ2S* x, int64_t nb,
                       const uint64_t* grid, const uint8_t* kmask) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * QK_K;
  const float d = h2f(x[i].d);
  const uint8_t* qs = x[i].qs;
  const uint8_t* qh = x[i].qh;
  const uint8_t* signs = x[i].qs + QK_K / 8;
  int o = 0;
  for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
    const float db0 = d * (0.5f + float(x[i].scales[ib32] & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + float(x[i].scales[ib32] >> 4)) * 0.25f;
    for (int l = 0; l < 4; ++l) {
      const float dl = (l / 2) ? db1 : db0;
      const uint64_t g = grid[qs[l] | ((uint32_t(qh[ib32]) << (8 - 2 * l)) & 0x300)];
      const uint8_t* gp = reinterpret_cast<const uint8_t*>(&g);
      for (int j = 0; j < 8; ++j)
        put(y, o + j, dl * float(gp[j]) * ((signs[l] & kmask[j]) ? -1.f : 1.f));
      o += 8;
    }
    qs += 4; signs += 4;
  }
}

template <class Out>
__global__ void k_iq3xxs(Out* out, const BIQ3XXS* x, int64_t nb,
                         const uint32_t* grid, const uint8_t* ksigns, const uint8_t* kmask) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * QK_K;
  const float d = h2f(x[i].d);
  const uint8_t* qs = x[i].qs;
  const uint8_t* sas = qs + QK_K / 4;
  int o = 0;
  for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
    uint32_t aux32 = 0;
    for (int b = 0; b < 4; ++b) aux32 |= uint32_t(sas[4 * ib32 + b]) << (8 * b);
    const float db = d * (0.5f + float(aux32 >> 28)) * 0.5f;
    for (int l = 0; l < 4; ++l) {
      const uint8_t signs = ksigns[(aux32 >> (7 * l)) & 127];
      const uint32_t g1 = grid[qs[2 * l + 0]], g2 = grid[qs[2 * l + 1]];
      const uint8_t* p1 = reinterpret_cast<const uint8_t*>(&g1);
      const uint8_t* p2 = reinterpret_cast<const uint8_t*>(&g2);
      for (int j = 0; j < 4; ++j) {
        put(y, o + j,     db * float(p1[j]) * ((signs & kmask[j])     ? -1.f : 1.f));
        put(y, o + j + 4, db * float(p2[j]) * ((signs & kmask[j + 4]) ? -1.f : 1.f));
      }
      o += 8;
    }
    qs += 8;
  }
}

template <class Out>
__global__ void k_iq3s(Out* out, const BIQ3S* x, int64_t nb,
                       const uint32_t* grid, const uint8_t* kmask) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nb) return;
  Out* y = out + i * QK_K;
  const float d = h2f(x[i].d);
  const uint8_t* qs = x[i].qs;
  const uint8_t* qh = x[i].qh;
  const uint8_t* signs = x[i].signs;
  int o = 0;
  for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
    const float db1 = d * (1.f + 2.f * float(x[i].scales[ib32 / 2] & 0xF));
    const float db2 = d * (1.f + 2.f * float(x[i].scales[ib32 / 2] >> 4));
    for (int l = 0; l < 4; ++l) {
      const uint32_t g1 = grid[qs[2 * l + 0] | ((uint32_t(qh[0]) << (8 - 2 * l)) & 256)];
      const uint32_t g2 = grid[qs[2 * l + 1] | ((uint32_t(qh[0]) << (7 - 2 * l)) & 256)];
      const uint8_t* p1 = reinterpret_cast<const uint8_t*>(&g1);
      const uint8_t* p2 = reinterpret_cast<const uint8_t*>(&g2);
      for (int j = 0; j < 4; ++j) {
        put(y, o + j,     db1 * float(p1[j]) * ((signs[l] & kmask[j])     ? -1.f : 1.f));
        put(y, o + j + 4, db1 * float(p2[j]) * ((signs[l] & kmask[j + 4]) ? -1.f : 1.f));
      }
      o += 8;
    }
    qs += 8; signs += 4;
    for (int l = 0; l < 4; ++l) {
      const uint32_t g1 = grid[qs[2 * l + 0] | ((uint32_t(qh[1]) << (8 - 2 * l)) & 256)];
      const uint32_t g2 = grid[qs[2 * l + 1] | ((uint32_t(qh[1]) << (7 - 2 * l)) & 256)];
      const uint8_t* p1 = reinterpret_cast<const uint8_t*>(&g1);
      const uint8_t* p2 = reinterpret_cast<const uint8_t*>(&g2);
      for (int j = 0; j < 4; ++j) {
        put(y, o + j,     db2 * float(p1[j]) * ((signs[l] & kmask[j])     ? -1.f : 1.f));
        put(y, o + j + 4, db2 * float(p2[j]) * ((signs[l] & kmask[j + 4]) ? -1.f : 1.f));
      }
      o += 8;
    }
    qh += 2; qs += 8; signs += 4;
  }
}

template <class Out>
__global__ void k_f32(Out* out, const float* x, int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) put(out + i, 0, x[i]);
}
template <class Out>
__global__ void k_f16(Out* out, const uint16_t* x, int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) put(out + i, 0, h2f(x[i]));
}
template <class Out>
__global__ void k_bf16(Out* out, const uint16_t* x, int64_t n) {
  const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint32_t u = uint32_t(x[i]) << 16;
    float f; memcpy(&f, &u, 4);
    put(out + i, 0, f);
  }
}

// Device copies of the vendored tables.
struct Tables {
  int8_t* kv4nl = nullptr;
  uint64_t* iq2xs = nullptr;
  uint64_t* iq2xxs = nullptr;
  uint64_t* iq2s = nullptr;
  uint32_t* iq3xxs = nullptr;
  uint32_t* iq3s = nullptr;
  uint8_t* ksigns = nullptr;
  uint8_t* kmask = nullptr;
};
Tables g_tab;

#define CKD(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); abort(); } } while(0)

template <class T>
T* upload(const T* h, size_t n) {
  T* d = nullptr;
  CKD(cudaMalloc(&d, n * sizeof(T)));
  CKD(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

void ensure_tables() {
  if (g_tab.kv4nl) return;
  g_tab.kv4nl  = upload(kvalues_iq4nl, 16);
  g_tab.iq2xxs = upload(iq2xxs_grid, 256);
  g_tab.iq2xs  = upload(iq2xs_grid, 512);
  g_tab.iq2s   = upload(iq2s_grid, 1024);
  g_tab.iq3xxs = upload(iq3xxs_grid, 256);
  g_tab.iq3s   = upload(iq3s_grid, 512);
  g_tab.ksigns = upload(ksigns_iq2xs, 128);
  g_tab.kmask  = upload(kmask_iq2xs, 8);
}

template <class Out>
void dispatch(Out* dst, const void* src, GgmlType t, int64_t n, cudaStream_t st) {
  ensure_tables();
  const int be = ggml_block_elems(t);
  if (n % be != 0) { fprintf(stderr, "gguf dequant: n=%lld not a multiple of block %d\n",
                             (long long)n, be); abort(); }
  const int64_t nb = n / be;
  const int TH = 128;
  const int64_t G = (nb + TH - 1) / TH;
  const int64_t Gn = (n + TH - 1) / TH;
  switch (t) {
    case GgmlType::F32:  k_f32<<<Gn, TH, 0, st>>>(dst, (const float*)src, n); break;
    case GgmlType::F16:  k_f16<<<Gn, TH, 0, st>>>(dst, (const uint16_t*)src, n); break;
    case GgmlType::BF16: k_bf16<<<Gn, TH, 0, st>>>(dst, (const uint16_t*)src, n); break;
    case GgmlType::Q8_0: k_q80<<<G, TH, 0, st>>>(dst, (const BQ80*)src, nb); break;
    case GgmlType::Q2_K: k_q2k<<<G, TH, 0, st>>>(dst, (const BQ2K*)src, nb); break;
    case GgmlType::Q3_K: k_q3k<<<G, TH, 0, st>>>(dst, (const BQ3K*)src, nb); break;
    case GgmlType::Q4_K: k_q4k<<<G, TH, 0, st>>>(dst, (const BQ4K*)src, nb); break;
    case GgmlType::Q5_K: k_q5k<<<G, TH, 0, st>>>(dst, (const BQ5K*)src, nb); break;
    case GgmlType::Q6_K: k_q6k<<<G, TH, 0, st>>>(dst, (const BQ6K*)src, nb); break;
    case GgmlType::IQ4_NL: k_iq4nl<<<G, TH, 0, st>>>(dst, (const BIQ4NL*)src, nb, g_tab.kv4nl); break;
    case GgmlType::IQ4_XS: k_iq4xs<<<G, TH, 0, st>>>(dst, (const BIQ4XS*)src, nb, g_tab.kv4nl); break;
    case GgmlType::IQ2_XS: k_iq2xs<<<G, TH, 0, st>>>(dst, (const BIQ2XS*)src, nb,
                                                     g_tab.iq2xs, g_tab.ksigns, g_tab.kmask); break;
    case GgmlType::IQ2_XXS: k_iq2xxs<<<G, TH, 0, st>>>(dst, (const BIQ2XXS*)src, nb,
                                                       g_tab.iq2xxs, g_tab.ksigns, g_tab.kmask); break;
    case GgmlType::IQ2_S: k_iq2s<<<G, TH, 0, st>>>(dst, (const BIQ2S*)src, nb,
                                                   g_tab.iq2s, g_tab.kmask); break;
    case GgmlType::IQ3_XXS: k_iq3xxs<<<G, TH, 0, st>>>(dst, (const BIQ3XXS*)src, nb,
                                                       g_tab.iq3xxs, g_tab.ksigns, g_tab.kmask); break;
    case GgmlType::IQ3_S: k_iq3s<<<G, TH, 0, st>>>(dst, (const BIQ3S*)src, nb,
                                                   g_tab.iq3s, g_tab.kmask); break;
    default:
      fprintf(stderr, "gguf dequant: type %s is not implemented\n", ggml_type_name(t));
      abort();
  }
}

}  // namespace

bool gguf_dequant_supported(GgmlType t) {
  switch (t) {
    case GgmlType::F32: case GgmlType::F16: case GgmlType::BF16:
    case GgmlType::Q8_0: case GgmlType::Q2_K: case GgmlType::Q3_K:
    case GgmlType::Q4_K: case GgmlType::Q5_K: case GgmlType::Q6_K:
    case GgmlType::IQ4_NL: case GgmlType::IQ4_XS: case GgmlType::IQ2_XS:
    case GgmlType::IQ2_S: case GgmlType::IQ2_XXS:
    case GgmlType::IQ3_XXS: case GgmlType::IQ3_S:
      return true;
    default: return false;
  }
}

void gguf_dequant_bf16(__nv_bfloat16* dst, const void* src, GgmlType t, int64_t n,
                       cudaStream_t st) { dispatch(dst, src, t, n, st); }
void gguf_dequant_f32(float* dst, const void* src, GgmlType t, int64_t n,
                      cudaStream_t st) { dispatch(dst, src, t, n, st); }

}  // namespace qwen
