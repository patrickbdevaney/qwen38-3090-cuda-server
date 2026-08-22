#include "gemv.cuh"
#include "ggml_tables.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <type_traits>

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
// AFFINE SPLIT.
//
// Every format here is affine: a value is `dl * code - ml` for a small integer
// code and a per-sub-block scale. Applying dl inside the dequantiser costs one
// multiply PER ELEMENT; hoisting it out costs one per eight, because
//     sum_i (dl * q_i) * x_i  ==  dl * sum_i q_i x_i.
// Types that declare AFFINE fill y[] with raw codes and return dl, and the
// kernel scales once per group. The summation order changes, so the dot product
// is not bit-identical -- it is a slightly more accurate association -- but the
// per-value dequantisation the gate checks is untouched, because get8() still
// applies the scale itself.
//
// Only the ml == 0 formats gain. Where a minimum is subtracted, `dl*q - ml` is
// already a single FMA per element and factoring would need sum(x) as well,
// which costs exactly what it saves.
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
    // The sub-block selects a byte offset, a nibble and a scale -- none of which
    // depend on i, so they come out of the loop, and what is left is eight
    // consecutive bytes of ql and eight of qh. A Q6_K block is 210 bytes, so
    // 64-bit loads are illegal and 16-bit ones are not: sixteen byte loads
    // become eight.
    const uint8_t* qlp = ql + l + ((sub & 1) ? 32 : 0);
    const int lsh = (sub >= 2) ? 4 : 0;
    const int hsh = 2 * sub;
    const float ds = d * float(sc[is + 2 * sub]);
    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(qlp);
    const uint16_t* h16 = reinterpret_cast<const uint16_t*>(qh + l);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t qv = q16[i], hv = h16[i];
      const int q0 = int((qv >> lsh) & 0xF)       | int(((hv >> hsh) & 3) << 4);
      const int q1 = int((qv >> (8 + lsh)) & 0xF) | int(((hv >> (8 + hsh)) & 3) << 4);
      y[2 * i]     = ds * float(q0 - 32);
      y[2 * i + 1] = ds * float(q1 - 32);
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
    // 16-bit loads, not 8-bit. A Q3_K block is 110 bytes, so consecutive blocks
    // land on alternating 8- and 4-byte alignments and a 64-bit load would be
    // illegal -- but 110 is even and every field offset here is a multiple of 8,
    // so a 16-bit load is always legal. That halves the memory instructions in
    // the hottest part of this path: sixteen byte loads (eight of qs, eight of
    // hmask) become eight. Q3_K was the slowest K-quant measured, and it is the
    // instruction count rather than the bytes that made it so.
    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(b->qs + n * 32 + 16 * half + l);
    const uint16_t* h16 = reinterpret_cast<const uint16_t*>(b->hmask + 16 * half + l);
    const uint32_t mh = uint32_t(m) << 8;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t qv = q16[i], hv = h16[i];
      y[2 * i]     = dl * (float(int((qv >> shift) & 3))       - ((hv & m)  ? 0.f : 4.f));
      y[2 * i + 1] = dl * (float(int((qv >> (8 + shift)) & 3)) - ((hv & mh) ? 0.f : 4.f));
    }
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
    // 16-bit loads: a Q2_K block is 84 bytes so 64-bit is illegal, but every
    // offset here is a multiple of 8 and 84 is even, so 16-bit always holds.
    const uint16_t* q16 = reinterpret_cast<const uint16_t*>(b->qs + n * 32 + 16 * half + l);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t qv = q16[i];
      y[2 * i]     = dl * float((qv >> shift) & 3) - ml;
      y[2 * i + 1] = dl * float((qv >> (8 + shift)) & 3) - ml;
    }
  }
};

// A 16-entry int8 lookup held in FOUR REGISTERS instead of memory.
//
// IQ4_XS and IQ4_NL map every 4-bit code through kvalues_iq4nl, which was eight
// loads per lane per 256-element run. Staging that table in shared memory bought
// nothing measurable, and the reason is bank conflicts: sixteen bytes read by
// thirty-two lanes at thirty-two different offsets collide on almost every
// access, so an LDS costs what the L1 hit it replaced cost. PRMT selects a byte
// out of an eight-byte register pair in one instruction, so two PRMTs and a
// per-byte blend replace four lookups with no memory traffic at all.
//
// `idx4` carries four 4-bit codes, one per byte, and the result carries the four
// table values the same way.
__device__ __forceinline__ uint32_t lut16_x4(uint32_t idx4) {
  // kvalues_iq4nl, packed four signed bytes to a word.
  constexpr uint32_t T0 = 0xBFAD9881u;   // -127 -104  -83  -65
  constexpr uint32_t T1 = 0xF6EADDCFu;   //  -49  -35  -22  -10
  constexpr uint32_t T2 = 0x26190D01u;   //    1   13   25   38
  constexpr uint32_t T3 = 0x71594535u;   //   53   69   89  113
  // PRMT's selector is four NIBBLES; the codes arrive as four bytes.
  const uint32_t b = idx4 & 0x07070707u;
  const uint32_t sel = (b & 0x7u) | ((b >> 4) & 0x70u) |
                       ((b >> 8) & 0x700u) | ((b >> 12) & 0x7000u);
  const uint32_t lo = __byte_perm(T0, T1, sel);          // codes 0-7
  const uint32_t hi = __byte_perm(T2, T3, sel);          // codes 8-15
  const uint32_t m  = __vcmpgeu4(idx4, 0x08080808u);     // 0xFF per byte with code >= 8
  return (lo & ~m) | (hi & m);
}

__device__ __forceinline__ float s8(uint32_t packed, int k) {
  return float(int(int8_t(packed >> (8 * k))));
}

template <> struct Deq<GgmlType::IQ4_XS> {
  static constexpr int RUN_BYTES = 136;
  static constexpr bool AFFINE = true;
  __device__ static float get8a(const uint8_t* p, int o0, float* y, const Tab& t) {
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
    const uint32_t p0 = lut16_x4((qv.x >> lsh) & 0x0F0F0F0Fu);
    const uint32_t p1 = lut16_x4((qv.y >> lsh) & 0x0F0F0F0Fu);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      y[i]     = s8(p0, i);
      y[i + 4] = s8(p1, i);
    }
    return dl;
  }
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab& t) {
    const float d_ = get8a(p, o0, y, t);
    #pragma unroll
    for (int i = 0; i < 8; ++i) y[i] *= d_;
  }
  // lut16_x4 already produces four codebook values packed as signed bytes,
  // which is exactly __dp4a's operand format, so the int8 path costs nothing
  // extra here -- it only SKIPS the conversion to float and the eight FMAs.
  static constexpr bool DP4A = true;
  __device__ static float get8q(const uint8_t* p, int o0, uint32_t* q, const Tab&) {
    const BIQ4XS* b = reinterpret_cast<const BIQ4XS*>(p);
    const int ib = o0 / 32, within = o0 % 32;
    const int ls = ((b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                   (((b->scales_h >> (2 * ib)) & 3) << 4);
    const uint2 qv = *reinterpret_cast<const uint2*>(b->qs + ib * 16 + (within % 16));
    const int lsh = (within >= 16) ? 4 : 0;
    q[0] = lut16_x4((qv.x >> lsh) & 0x0F0F0F0Fu);
    q[1] = lut16_x4((qv.y >> lsh) & 0x0F0F0F0Fu);
    return h2f(b->d) * float(ls - 32);
  }
};

// The i-quants all end the same way: eight grid bytes, a sign mask, one scale.
// Folding the sign into the INTEGER before the conversion is bit-exact -- an
// int negation and a float negation of the same magnitude agree exactly -- and
// it drops one multiply per element. Taking the bytes with shifts rather than
// through a pointer into a local also keeps the grid word in a register.
__device__ __forceinline__ void spread8(float* y, uint32_t g0, uint32_t g1,
                                        uint32_t sg, float db) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int a = int((g0 >> (8 * i)) & 0xFF);
    const int b = int((g1 >> (8 * i)) & 0xFF);
    y[i]     = db * float((sg >> i)       & 1 ? -a : a);
    y[i + 4] = db * float((sg >> (i + 4)) & 1 ? -b : b);
  }
}

template <> struct Deq<GgmlType::IQ2_XS> {
  static constexpr int RUN_BYTES = 74;
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab& t) {
    const BIQ2XS* b = reinterpret_cast<const BIQ2XS*>(p);
    const int ib32 = o0 / 32, l = (o0 % 32) / 8;
    const uint16_t qi = b->qs[4 * ib32 + l];
    const float db = h2f(b->d) *
        (0.5f + float((b->scales[ib32] >> (4 * (l / 2))) & 0xF)) * 0.25f;
    const uint64_t g = t.iq2xs[qi & 511];
    spread8(y, uint32_t(g), uint32_t(g >> 32), t.ksigns[qi >> 9], db);
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
    spread8(y, uint32_t(g), uint32_t(g >> 32), t.ksigns[(a1 >> (7 * l)) & 127], db);
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
    spread8(y, uint32_t(g), uint32_t(g >> 32), b->qs[32 + 4 * ib32 + l], db);
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
    // qs sits at offset 2 of a 98-byte block and the lane offset is even, so
    // the two grid indices come out of one 16-bit load.
    const uint32_t q2 = *reinterpret_cast<const uint16_t*>(b->qs + 8 * ib32 + 2 * l);
    spread8(y, t.iq3xxs[q2 & 0xFF], t.iq3xxs[q2 >> 8], sg, db);
  }
};

template <> struct Deq<GgmlType::IQ3_S> {
  static constexpr int RUN_BYTES = 110;
  static constexpr bool AFFINE = true;
  __device__ static float get8a(const uint8_t* p, int o0, float* y, const Tab& t) {
    const BIQ3S* b = reinterpret_cast<const BIQ3S*>(p);
    const int s = o0 / 32, l = (o0 % 32) / 8;
    const float db = h2f(b->d) *
        (1.f + 2.f * float((b->scales[s / 2] >> (4 * (s % 2))) & 0xF));
    const uint8_t qh = b->qh[s];
    // One 16-bit load for the two grid indices instead of two byte loads. qs
    // sits at offset 2 of a 110-byte block and the lane offset is even, so the
    // address is 2-aligned in every block.
    const uint32_t q2 = *reinterpret_cast<const uint16_t*>(b->qs + 8 * s + 2 * l);
    const uint32_t g1 = t.iq3s[(q2 & 0xFF) | ((uint32_t(qh) << (8 - 2 * l)) & 256)];
    const uint32_t g2 = t.iq3s[(q2 >> 8)   | ((uint32_t(qh) << (7 - 2 * l)) & 256)];
    const uint32_t sg = b->signs[4 * s + l];
    // The sign folds into the integer before the conversion: negating an int and
    // negating the float it converts to are the same value, so this is bit-exact
    // and drops a multiply per element.
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int v1 = int((g1 >> (8 * i)) & 0xFF);
      const int v2 = int((g2 >> (8 * i)) & 0xFF);
      y[i]     = float((sg >> i)       & 1 ? -v1 : v1);
      y[i + 4] = float((sg >> (i + 4)) & 1 ? -v2 : v2);
    }
    return db;
  }
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab& t) {
    const float d_ = get8a(p, o0, y, t);
    #pragma unroll
    for (int i = 0; i < 8; ++i) y[i] *= d_;
  }
  // MEASURED AND REJECTED for IQ3_S: 617 -> 550 GB/s. Unlike IQ4_XS, whose
  // codebook lookup already yields four signed bytes in a word, this format has
  // to negate each grid entry by its sign bit and then pack four of them --
  // which costs more than the eight float conversions and FMAs it removes. Kept
  // here because the next person will have the same idea.
  __device__ static float get8q_unused(const uint8_t* p, int o0, uint32_t* q, const Tab& t) {
    const BIQ3S* b = reinterpret_cast<const BIQ3S*>(p);
    const int sb = o0 / 32, l = (o0 % 32) / 8;
    const float db = h2f(b->d) *
        (1.f + 2.f * float((b->scales[sb / 2] >> (4 * (sb % 2))) & 0xF));
    const uint8_t qh = b->qh[sb];
    const uint32_t q2 = *reinterpret_cast<const uint16_t*>(b->qs + 8 * sb + 2 * l);
    const uint32_t g1 = t.iq3s[(q2 & 0xFF) | ((uint32_t(qh) << (8 - 2 * l)) & 256)];
    const uint32_t g2 = t.iq3s[(q2 >> 8)   | ((uint32_t(qh) << (7 - 2 * l)) & 256)];
    const uint32_t sg = b->signs[4 * sb + l];
    uint32_t a = 0, c = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int v1 = int((g1 >> (8 * i)) & 0xFF);
      const int v2 = int((g2 >> (8 * i)) & 0xFF);
      a |= uint32_t(uint8_t(int8_t((sg >> i)       & 1 ? -v1 : v1))) << (8 * i);
      c |= uint32_t(uint8_t(int8_t((sg >> (i + 4)) & 1 ? -v2 : v2))) << (8 * i);
    }
    q[0] = a; q[1] = c;
    return db;
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
  static constexpr bool AFFINE = true;
  __device__ static float get8a(const uint8_t* p, int o0, float* y, const Tab& t) {
    const BIQ4NL* b = reinterpret_cast<const BIQ4NL*>(p) + (o0 / 32);
    const float d = h2f(b->d);
    const int w = o0 % 32;
    const uint8_t* q = b->qs + (w % 16);
    const int lsh = (w >= 16) ? 4 : 0;
    // An IQ4_NL block is 18 bytes, so consecutive blocks land on rotating
    // alignments and these eight bytes cannot be one 64-bit load without a
    // repack. The table lookups still go through PRMT.
    uint32_t w0 = 0, w1 = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      w0 |= uint32_t(q[i]) << (8 * i);
      w1 |= uint32_t(q[i + 4]) << (8 * i);
    }
    const uint32_t p0 = lut16_x4((w0 >> lsh) & 0x0F0F0F0Fu);
    const uint32_t p1 = lut16_x4((w1 >> lsh) & 0x0F0F0F0Fu);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      y[i]     = s8(p0, i);
      y[i + 4] = s8(p1, i);
    }
    return d;
  }
  __device__ static void get8(const uint8_t* p, int o0, float* y, const Tab& t) {
    const float d_ = get8a(p, o0, y, t);
    #pragma unroll
    for (int i = 0; i < 8; ++i) y[i] *= d_;
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
#ifndef QWEN_GGUF_WARPS
#define QWEN_GGUF_WARPS 2
#endif
constexpr int GEMV_WARPS = QWEN_GGUF_WARPS;
constexpr int GEMV_THREADS = GEMV_WARPS * 32;

// ---------------------------------------------------------------- codebooks
//
// The i-quants are CODEBOOK quants: the stored index selects an entry from a
// fixed grid. Those grids live in global memory and were being indexed once per
// value -- eight dependent global loads per lane per 256-run, on top of the
// weight stream itself. They are tiny (1-8 KiB) and every warp in the block
// hits the same entries, so staging them in shared memory once per block turns
// each lookup into an LDS. This is the largest remaining share of the gap
// against llama.cpp: IQ4_XS and IQ3_S together are 65% of UD-Q3_K_XL's bytes
// and were the two slowest types in the sweep.
//
// Types with no codebook stage nothing and pay one predicated branch.
template <GgmlType T> struct Stage {
  static constexpr int BYTES = 1;   // zero-size __shared__ arrays are illegal
  __device__ static Tab go(uint8_t*, Tab t, int) { return t; }
};

// Copy `n` bytes cooperatively and hand back a pointer into shared.
__device__ __forceinline__ void* sm_copy(uint8_t*& sp, const void* src, int n, int tid,
                                         int nthreads) {
  uint8_t* dst = sp;
  const uint8_t* s8 = static_cast<const uint8_t*>(src);
  for (int i = tid * 4; i < n; i += nthreads * 4)
    *reinterpret_cast<uint32_t*>(dst + i) = *reinterpret_cast<const uint32_t*>(s8 + i);
  sp += n;
  return dst;
}

#define QWEN_STAGE(TYPE, BYTES_, BODY)                                        \
  template <> struct Stage<GgmlType::TYPE> {                                  \
    static constexpr int BYTES = BYTES_;                                      \
    __device__ static Tab go(uint8_t* sp, Tab t, int tid) { BODY return t; }  \
  };

// kvalues_iq4nl is 16 bytes; it is read eight times per lane per run.
// IQ4_XS and IQ4_NL stage nothing: lut16_x4 keeps their table in registers.
QWEN_STAGE(IQ3_S, 512 * 4,
  t.iq3s = (const uint32_t*)sm_copy(sp, t.iq3s, 512 * 4, tid, GEMV_THREADS);)
QWEN_STAGE(IQ3_XXS, 256 * 4 + 128,
  t.iq3xxs = (const uint32_t*)sm_copy(sp, t.iq3xxs, 256 * 4, tid, GEMV_THREADS);
  t.ksigns = (const uint8_t*)sm_copy(sp, t.ksigns, 128, tid, GEMV_THREADS);)
QWEN_STAGE(IQ2_XS, 512 * 8 + 128,
  t.iq2xs = (const uint64_t*)sm_copy(sp, t.iq2xs, 512 * 8, tid, GEMV_THREADS);
  t.ksigns = (const uint8_t*)sm_copy(sp, t.ksigns, 128, tid, GEMV_THREADS);)
QWEN_STAGE(IQ2_XXS, 256 * 8 + 128,
  t.iq2xxs = (const uint64_t*)sm_copy(sp, t.iq2xxs, 256 * 8, tid, GEMV_THREADS);
  t.ksigns = (const uint8_t*)sm_copy(sp, t.ksigns, 128, tid, GEMV_THREADS);)
// IQ2_S is deliberately NOT staged. Its grid is 8 KiB -- four times the next
// biggest -- and measured warm, staging it cost 4.4% (270.0 -> 258.1 GB/s)
// where IQ3_S gained 12%. The table is big enough that the occupancy it takes
// costs more than the loads it saves.
#undef QWEN_STAGE

// bf16 -> float is a 16-bit left shift; a packed pair is two masks.
__device__ __forceinline__ float xlo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float xhi(uint32_t u) { return __uint_as_float(u & 0xFFFF0000u); }

// Types without an AFFINE member are detected, not annotated, so adding the
// split to another format is a two-line change to that format alone.
template <class D, class = void> struct HasAffine : std::false_type {};
template <class D> struct HasAffine<D, decltype((void)D::AFFINE)> : std::true_type {};
template <class D, class = void> struct HasDp4a : std::false_type {};
template <class D> struct HasDp4a<D, decltype((void)D::DP4A)> : std::true_type {};

// One block per 32-element group per row. absmax -> scale, round to int8.
__global__ void k_quant_x(int8_t* __restrict__ qx, float* __restrict__ xsc,
                          const __nv_bfloat16* __restrict__ x, int in_f) {
  const int g = blockIdx.x;                 // group index within the flattened [M, in_f]
  const size_t base = size_t(g) * 32;
  const int lane = threadIdx.x;             // 32 threads
  const float v = __bfloat162float(x[base + lane]);
  float a = fabsf(v);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
  const float sc = a > 0.f ? a / 127.0f : 1.0f;
  qx[base + lane] = int8_t(__float2int_rn(v / sc));
  if (lane == 0) xsc[g] = sc;
}

template <GgmlType T, int MROWS>
__global__ __launch_bounds__(GEMV_THREADS) void k_gemv(
    __nv_bfloat16* __restrict__ y, const uint8_t* __restrict__ w,
    const __nv_bfloat16* __restrict__ x, int out_f, int in_f, int ldy, Tab tab,
    const int8_t* __restrict__ qx, const float* __restrict__ xsc) {
  // Stage the codebook BEFORE the out-of-range return: every thread in the
  // block has to reach the barrier, and the tail block has threads that do not
  // own a row.
  __shared__ __align__(8) uint8_t sm[Stage<T>::BYTES];
  if (Stage<T>::BYTES > 1) {
    tab = Stage<T>::go(sm, tab, threadIdx.x);
    __syncthreads();
  }

  const int warp = (blockIdx.x * GEMV_WARPS) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (warp >= out_f) return;
  const int runs = in_f >> 8;
  const uint8_t* wr = w + size_t(warp) * size_t(runs) * Deq<T>::RUN_BYTES;
  const int o0 = lane * 8;

  float acc[MROWS];
  #pragma unroll
  for (int m = 0; m < MROWS; ++m) acc[m] = 0.f;

  // TWO runs in flight. Every measurable throughput bound says this kernel
  // should be three times faster than it is -- instruction issue, integer ALU
  // and DRAM all have headroom, occupancy is at the 48-warp ceiling, the FMA
  // chain is not exposed and the activations are L1-resident. What is left is
  // LATENCY: each lane's dequantiser is a short dependent chain of loads
  // (header, then scale, then data) and there is only one of them outstanding
  // per lane at a time. Decoding two super-blocks per iteration gives the
  // scheduler two independent chains to interleave.
  constexpr bool AF = HasAffine<Deq<T>>::value;
  // The int8 activation path is only taken when the caller supplied a quantised
  // vector; otherwise this type falls back to its float dequantiser.
  const bool DP = HasDp4a<Deq<T>>::value && qx != nullptr;
  auto body = [&](int i, const float* v, float dl) {
    const int base = (i << 8) + o0;
    #pragma unroll
    for (int m = 0; m < MROWS; ++m) {
      // ONE 16-byte load for the lane's eight activations, not eight 2-byte
      // ones. `base` is a multiple of 8 and in_f a multiple of 256, so the
      // address is always 16-byte aligned. bf16 -> float is a shift, so the
      // halves are unpacked with bit ops rather than through a local array,
      // which would spill the vector to local memory and undo the point.
      const uint4 xv = *reinterpret_cast<const uint4*>(x + size_t(m) * in_f + base);
      // With the affine split the scale multiplies the GROUP sum, not each
      // element: eight multiplies per eight values become one.
      float a = AF ? 0.f : acc[m];
      a = fmaf(v[0], xlo(xv.x), a);
      a = fmaf(v[1], xhi(xv.x), a);
      a = fmaf(v[2], xlo(xv.y), a);
      a = fmaf(v[3], xhi(xv.y), a);
      a = fmaf(v[4], xlo(xv.z), a);
      a = fmaf(v[5], xhi(xv.z), a);
      a = fmaf(v[6], xlo(xv.w), a);
      a = fmaf(v[7], xhi(xv.w), a);
      acc[m] = AF ? fmaf(dl, a, acc[m]) : a;
    }
  };

#ifndef QWEN_GGUF_UNROLL
#define QWEN_GGUF_UNROLL 4
#endif
  // Wide M already gives the scheduler independent work per row, and v[U][8]
  // costs 8U registers -- at U=4 with MROWS=8 that spilled. Swept: U=4 is the
  // best for the narrow shapes (2 -> 523, 3 -> 520, 4 -> 542, 6 -> 530 GB/s
  // composition-weighted), and warps per block barely matters (2 -> 553,
  // 4 -> 545, 8 -> 535), which is itself evidence the kernel is latency bound
  // rather than occupancy bound.
  // Also tried and rejected: -Xptxas=-dlcm=cg on this translation unit, to keep
  // the streamed weights out of L1 and leave it to the activation vector. No
  // change (578 vs 580 GB/s composition-weighted), so the default policy stays.
  constexpr int U = QWEN_GGUF_UNROLL;
  constexpr int UU = U;
  if constexpr (HasDp4a<Deq<T>>::value) {
    if (DP) {
      for (int i = 0; i < runs; ++i) {
        uint32_t qw[2];
        const float dl = Deq<T>::get8q(wr + size_t(i) * Deq<T>::RUN_BYTES, o0, qw, tab);
        const int base = (i << 8) + o0;
        #pragma unroll
        for (int m = 0; m < MROWS; ++m) {
          const size_t off = size_t(m) * in_f + base;
          const uint2 xv = *reinterpret_cast<const uint2*>(qx + off);
          int d = __dp4a(int(qw[0]), int(xv.x), 0);
          d = __dp4a(int(qw[1]), int(xv.y), d);
          // One group scale per 32 activations; a lane's eight elements always
          // sit inside one group.
          acc[m] = fmaf(dl * xsc[off >> 5], float(d), acc[m]);
        }
      }
      #pragma unroll
      for (int m = 0; m < MROWS; ++m) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o);
        if (lane == 0) y[size_t(m) * ldy + warp] = __float2bfloat16(acc[m]);
      }
      return;
    }
  }
  int i = 0;
  for (; i + UU <= runs; i += UU) {
    float v[UU][8], dl[UU];
    #pragma unroll
    for (int u = 0; u < UU; ++u) {
      if constexpr (AF)
        dl[u] = Deq<T>::get8a(wr + size_t(i + u) * Deq<T>::RUN_BYTES, o0, v[u], tab);
      else { dl[u] = 1.f; Deq<T>::get8(wr + size_t(i + u) * Deq<T>::RUN_BYTES, o0, v[u], tab); }
    }
    #pragma unroll
    for (int u = 0; u < UU; ++u) body(i + u, v[u], dl[u]);
  }
  for (; i < runs; ++i) {
    float v[8], d1 = 1.f;
    if constexpr (AF) d1 = Deq<T>::get8a(wr + size_t(i) * Deq<T>::RUN_BYTES, o0, v, tab);
    else Deq<T>::get8(wr + size_t(i) * Deq<T>::RUN_BYTES, o0, v, tab);
    body(i, v, d1);
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
            int ldy, cudaStream_t st, const int8_t* qx, const float* xsc) {
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
    case N: k_gemv<T, N><<<blocks, GEMV_THREADS, 0, st>>>(y, p, x, w.out_f, w.in_f, ldy, tb, qx, xsc); break;
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

void gguf_quantize_x(int8_t* qx, float* xsc, const __nv_bfloat16* x, int in_f, int M,
                     cudaStream_t st) {
  k_quant_x<<<(in_f / 32) * M, 32, 0, st>>>(qx, xsc, x, in_f);
}

bool gguf_wants_qx(GgmlType t) {
  // Kill switch, so the int8 path can be A/B'd against the float one on the
  // same binary and the same prompts.
  static int off = -1;
  if (off < 0) { const char* e = getenv("QWEN_GGUF_NO_DP4A"); off = e && atoi(e); }
  if (off) return false;
  // IQ3_S implements the int8 path too, but measured slower with it -- see the
  // note on Deq<IQ3_S>::get8q_unused.
  return t == GgmlType::IQ4_XS;
}

void gguf_gemv(__nv_bfloat16* y, const GgufWeight& w, const __nv_bfloat16* x,
               cudaStream_t st, int ldy, const int8_t* qx, const float* xsc) {
  if (w.in_f % 256) { fprintf(stderr, "gguf gemv: in_f %d is not a multiple of 256\n",
                              w.in_f); abort(); }
  check_align(w, "gguf_gemv");
  DISPATCH(launch, y, w, x, 1, ldy, st, qx, xsc);
}

void gguf_gemm_small(__nv_bfloat16* y, const GgufWeight& w, const __nv_bfloat16* x,
                     int M, cudaStream_t st, int ldy, const int8_t* qx, const float* xsc) {
  if (w.in_f % 256) { fprintf(stderr, "gguf gemm: in_f %d is not a multiple of 256\n",
                              w.in_f); abort(); }
  check_align(w, "gguf_gemm_small");
  DISPATCH(launch, y, w, x, M, ldy, st, qx, xsc);
}

}  // namespace qwen
