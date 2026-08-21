// compressed-tensors "pack-quantized" int4 unpack.
//
// Format, taken from compressed_tensors/compressors/pack_quantized/helpers.py
// and pinned by tests/gate_dequant.cpp against the library's own output:
//
//   * a weight is int8 in [-8, 7], stored as the nibble  n = w + 8  in [0, 15]
//   * packing is DENSE and little-endian along the packed dimension: element i
//     occupies global bits [4i, 4i+4). Because 4 divides 32, element e of a row
//     lives in word (e/8) at bit 4*(e%8) and never straddles a word boundary --
//     which is the whole reason a fused dequant-in-the-load GEMV is cheap here.
//   * weight_packed is packed along dim 1 (input features):  [out, in/8] i32
//   * weight_zero_point is packed along dim 0 (output features): [out/8, G] i32
//     and carries the SAME -8 offset
//   * dequant is  w[o][i] = (q[o][i] - zp[o][g]) * scale[o][g],  g = i/group
//
// Getting the zero-point axis backwards is the easy mistake: its packed shape
// [out/8, G] looks like a row-major [rows, cols] but the packing runs down the
// OUTPUT axis, so zp for row o is nibble (o%8) of word [o/8][g].
#pragma once
#include <cstdint>

namespace qwen {

// Nibble e of a row of packed words, returned as signed int in [-8, 7].
inline int32_t unpack_nibble(const uint32_t* words, int64_t e) {
  const uint32_t w = words[e >> 3];
  const int      s = static_cast<int>((e & 7) * 4);
  return static_cast<int32_t>((w >> s) & 0xFu) - 8;
}

// q[o][i] for a weight_packed tensor of shape [out, in/8].
inline int32_t w4_q(const uint32_t* packed, int64_t in_features, int64_t o, int64_t i) {
  return unpack_nibble(packed + o * (in_features >> 3), i);
}

// zero_point[o][g] for a weight_zero_point tensor of shape [out/8, G].
// Packed down the OUTPUT axis: row o is nibble (o % 8) of word [o/8][g].
inline int32_t w4_zp(const uint32_t* zp, int64_t num_groups, int64_t o, int64_t g) {
  const uint32_t w = zp[(o >> 3) * num_groups + g];
  const int      s = static_cast<int>((o & 7) * 4);
  return static_cast<int32_t>((w >> s) & 0xFu) - 8;
}

}  // namespace qwen
