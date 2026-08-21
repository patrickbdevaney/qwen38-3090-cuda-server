// Device dequantisation for the ggml block formats a UD-IQ4_XS file contains.
//
// "UD-IQ4_XS" names the target, not the contents: unsloth's dynamic quants pick
// a type per tensor, and this file measures eleven of them --
//   IQ4_XS 50.7%, Q5_K 14.7%, IQ3_S 11.3%, Q4_K 8.9%, Q3_K 5.6%,
//   IQ3_XXS 3.8%, Q6_K 2.4%, and a tail of IQ2_XS / IQ4_NL / Q2_K / Q8_0.
// A loader that only handles the headline type cannot open the file at all.
//
// The K-quants are linear: a 6-bit sub-block scale (and for Q2_K/Q4_K/Q5_K a
// sub-block min) against an fp16 super-block scale. The i-quants are CODEBOOK
// quants -- the stored index selects an entry from a fixed grid, with a separate
// sign mask -- which is why src/gguf/ggml_tables.h exists.
//
// Every one of these is gated bit-for-bit against ggml's own dequantize_row_*
// via tools/gguf_dequant_ref.cpp. They were ported, not invented.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>
#include "gguf.h"

namespace qwen {

// Dequantise `n` elements of `type` from `src` (device, raw block bytes) into
// `dst` (device, bf16). Aborts on a type this build does not implement.
void gguf_dequant_bf16(__nv_bfloat16* dst, const void* src, GgmlType type,
                       int64_t n, cudaStream_t stream = 0);

// Same, to fp32 -- used by the gate so the comparison against ggml is not
// blurred by a bf16 round trip.
void gguf_dequant_f32(float* dst, const void* src, GgmlType type, int64_t n,
                      cudaStream_t stream = 0);

// True if this build can dequantise the type.
bool gguf_dequant_supported(GgmlType type);

}  // namespace qwen
