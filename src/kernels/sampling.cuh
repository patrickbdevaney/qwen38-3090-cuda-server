// On-GPU sampling.
//
// 248,320 logits per token. A single device-to-host copy of the logit vector
// would be 993 KB per token -- about 1 GB/s of PCIe traffic at target speeds --
// and would serialise the pipeline behind a synchronous copy. Only the chosen
// token id crosses the bus.
//
// Determinism: the greedy path is bitwise reproducible run to run (fixed
// reduction order, lowest index wins ties). The sampled path is reproducible for
// a given seed and step index, because the RNG is counter-based rather than
// stateful -- a stateful RNG would not survive CUDA graph replay.
#pragma once
#include <cuda_bf16.h>
#include <cstdint>

namespace qwen {

struct SamplingParams {
  float temperature = 1.0f;
  float top_p = 1.0f;
  int   top_k = 0;              // 0 = disabled
  float min_p = 0.0f;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  float repetition_penalty = 1.0f;
  uint64_t seed = 0;
};

struct SamplerState {
  float*   probs = nullptr;      // [vocab] scratch
  int32_t* idx = nullptr;        // [vocab] scratch
  float*   red = nullptr;        // reduction scratch
  int32_t* counts = nullptr;     // [vocab] token occurrence counts, for penalties
  int      vocab = 0;
};

void sampler_alloc(SamplerState& s, int vocab);
void sampler_free(SamplerState& s);
void sampler_reset_counts(SamplerState& s, cudaStream_t stream = 0);
// Records a committed token so the penalties see it.
void sampler_note_token(SamplerState& s, int32_t token, cudaStream_t stream = 0);

// Writes the chosen id to out_id (device). step is mixed into the RNG so a
// captured graph replays with different randomness each step.
void sample(int32_t* out_id, const __nv_bfloat16* logits, SamplerState& s,
            const SamplingParams& p, uint64_t step, cudaStream_t stream = 0);

}  // namespace qwen
