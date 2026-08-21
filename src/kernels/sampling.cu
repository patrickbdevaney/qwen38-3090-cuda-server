#include "sampling.cuh"
#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

namespace qwen {
namespace {

constexpr int CHUNKS = 512;
constexpr int THREADS = 256;

// Counter-based RNG (splitmix64). Stateless, so a CUDA graph replays correctly:
// a stateful generator would produce the same number every replay.
__device__ __forceinline__ float rng01(uint64_t seed, uint64_t step, uint64_t k) {
  uint64_t z = seed + step * 0x9E3779B97F4A7C15ull + k * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z = z ^ (z >> 31);
  return float(z >> 40) * (1.0f / 16777216.0f);   // [0,1)
}

// Pass 1: apply penalties and temperature, find the max.
__global__ void k_prep(float* __restrict__ out, const __nv_bfloat16* __restrict__ logits,
                       const int32_t* __restrict__ counts, int n, float inv_temp,
                       float pres, float freq, float rep, float* __restrict__ blockmax) {
  const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  float mx = -FLT_MAX;
  for (int i = i0; i < n; i += stride) {
    float v = __bfloat162float(logits[i]);
    const int c = counts ? counts[i] : 0;
    if (c > 0) {
      // OpenAI semantics: presence is a flat penalty for having appeared at all,
      // frequency scales with the count. repetition_penalty is the multiplicative
      // HF form, which divides positive logits and multiplies negative ones.
      v -= pres + freq * float(c);
      if (rep != 1.0f) v = (v > 0.f) ? v / rep : v * rep;
    }
    v *= inv_temp;
    out[i] = v;
    mx = fmaxf(mx, v);
  }
  __shared__ float sm[THREADS];
  sm[threadIdx.x] = mx;
  __syncthreads();
  for (int s = THREADS >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) blockmax[blockIdx.x] = sm[0];
}

__global__ void k_expsum(float* __restrict__ p, int n, const float* __restrict__ blockmax,
                         float* __restrict__ out_sum, float* __restrict__ out_max) {
  __shared__ float sm[THREADS];
  float mx = -FLT_MAX;
  for (int i = threadIdx.x; i < CHUNKS; i += THREADS) mx = fmaxf(mx, blockmax[i]);
  sm[threadIdx.x] = mx; __syncthreads();
  for (int s = THREADS >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
    __syncthreads();
  }
  const float m = sm[0];
  if (threadIdx.x == 0) *out_max = m;
  __syncthreads();
  float acc = 0.f;
  for (int i = threadIdx.x; i < n; i += THREADS) {
    const float e = __expf(p[i] - m);
    p[i] = e;
    acc += e;
  }
  sm[threadIdx.x] = acc; __syncthreads();
  for (int s = THREADS >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) *out_sum = sm[0];
}

// Single-block sampler over the (normalised) probabilities.
//
// top_k is applied by an iterative threshold search rather than a sort: bisect
// on the probability threshold until the number of survivors is <= k. A full
// 248k-element sort per token would cost far more than the model step.
__global__ void k_select(int32_t* __restrict__ out, float* __restrict__ p, int n,
                         const float* __restrict__ psum, float top_p, int top_k,
                         float min_p, uint64_t seed, uint64_t step, int greedy) {
  __shared__ float sm[THREADS];
  __shared__ float s_thresh, s_kept;
  const float total = *psum;

  if (greedy) {
    float best = -1.f; int bi = 0;
    for (int i = threadIdx.x; i < n; i += THREADS) {
      if (p[i] > best) { best = p[i]; bi = i; }
    }
    __shared__ float bv[THREADS]; __shared__ int bidx[THREADS];
    bv[threadIdx.x] = best; bidx[threadIdx.x] = bi;
    __syncthreads();
    for (int s = THREADS >> 1; s > 0; s >>= 1) {
      if (threadIdx.x < s) {
        const int j = threadIdx.x + s;
        if (bv[j] > bv[threadIdx.x] ||
            (bv[j] == bv[threadIdx.x] && bidx[j] < bidx[threadIdx.x])) {
          bv[threadIdx.x] = bv[j]; bidx[threadIdx.x] = bidx[j];
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = bidx[0];
    return;
  }

  // ---- min_p and top_k thresholds ----
  float pmax = 0.f;
  for (int i = threadIdx.x; i < n; i += THREADS) pmax = fmaxf(pmax, p[i]);
  sm[threadIdx.x] = pmax; __syncthreads();
  for (int s = THREADS >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + s]);
    __syncthreads();
  }
  float cut = (min_p > 0.f) ? sm[0] * min_p : 0.f;

  if (top_k > 0 && top_k < n) {
    float lo = 0.f, hi = sm[0];
    for (int it = 0; it < 24; ++it) {
      const float mid = 0.5f * (lo + hi);
      int cnt = 0;
      for (int i = threadIdx.x; i < n; i += THREADS) cnt += (p[i] >= mid);
      __shared__ int ci[THREADS];
      ci[threadIdx.x] = cnt; __syncthreads();
      for (int s = THREADS >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) ci[threadIdx.x] += ci[threadIdx.x + s];
        __syncthreads();
      }
      if (ci[0] > top_k) lo = mid; else hi = mid;
      __syncthreads();
    }
    cut = fmaxf(cut, lo);
  }

  // ---- surviving mass ----
  float kept = 0.f;
  for (int i = threadIdx.x; i < n; i += THREADS) if (p[i] >= cut) kept += p[i];
  sm[threadIdx.x] = kept; __syncthreads();
  for (int s = THREADS >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) { s_kept = sm[0]; s_thresh = cut; }
  __syncthreads();

  // ---- top_p: raise the threshold until the surviving mass is <= top_p ----
  float thresh = s_thresh, mass = s_kept;
  if (top_p < 1.0f) {
    float lo = s_thresh, hi = 1.0f;
    for (int it = 0; it < 24; ++it) {
      const float mid = 0.5f * (lo + hi);
      float acc = 0.f;
      for (int i = threadIdx.x; i < n; i += THREADS) if (p[i] >= mid) acc += p[i];
      sm[threadIdx.x] = acc; __syncthreads();
      for (int s = THREADS >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
      }
      // keep the smallest set whose mass still covers top_p
      if (sm[0] >= top_p * total) lo = mid; else hi = mid;
      __syncthreads();
    }
    thresh = lo;
    float acc = 0.f;
    for (int i = threadIdx.x; i < n; i += THREADS) if (p[i] >= thresh) acc += p[i];
    sm[threadIdx.x] = acc; __syncthreads();
    for (int s = THREADS >> 1; s > 0; s >>= 1) {
      if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
      __syncthreads();
    }
    mass = sm[0];
  }

  // ---- inverse-CDF draw over the survivors, in index order ----
  if (threadIdx.x == 0) {
    const float u = rng01(seed, step, 0) * mass;
    float c = 0.f;
    int pick = -1;
    for (int i = 0; i < n; ++i) {
      if (p[i] >= thresh) {
        c += p[i];
        if (c >= u) { pick = i; break; }
      }
    }
    if (pick < 0) { for (int i = n - 1; i >= 0; --i) if (p[i] >= thresh) { pick = i; break; } }
    out[0] = pick < 0 ? 0 : pick;
  }
}

__global__ void k_note(int32_t* counts, int32_t tok) { counts[tok] += 1; }

}  // namespace

void sampler_alloc(SamplerState& s, int vocab) {
  s.vocab = vocab;
  cudaMalloc(&s.probs, size_t(vocab) * 4);
  cudaMalloc(&s.idx, size_t(vocab) * 4);
  cudaMalloc(&s.red, size_t(CHUNKS + 8) * 4);
  cudaMalloc(&s.counts, size_t(vocab) * 4);
  cudaMemset(s.counts, 0, size_t(vocab) * 4);
}
void sampler_free(SamplerState& s) {
  cudaFree(s.probs); cudaFree(s.idx); cudaFree(s.red); cudaFree(s.counts);
  s = SamplerState{};
}
void sampler_reset_counts(SamplerState& s, cudaStream_t st) {
  cudaMemsetAsync(s.counts, 0, size_t(s.vocab) * 4, st);
}
void sampler_note_token(SamplerState& s, int32_t tok, cudaStream_t st) {
  k_note<<<1, 1, 0, st>>>(s.counts, tok);
}

void sample(int32_t* out, const __nv_bfloat16* logits, SamplerState& s,
            const SamplingParams& p, uint64_t step, cudaStream_t st) {
  const bool greedy = (p.temperature <= 0.f);
  const float inv_t = greedy ? 1.0f : 1.0f / p.temperature;
  const bool pen = (p.presence_penalty != 0.f || p.frequency_penalty != 0.f ||
                    p.repetition_penalty != 1.0f);
  float* blockmax = s.red;
  float* psum = s.red + CHUNKS;
  float* pmax = s.red + CHUNKS + 1;
  k_prep<<<CHUNKS, THREADS, 0, st>>>(s.probs, logits, pen ? s.counts : nullptr, s.vocab,
                                     inv_t, p.presence_penalty, p.frequency_penalty,
                                     p.repetition_penalty, blockmax);
  k_expsum<<<1, THREADS, 0, st>>>(s.probs, s.vocab, blockmax, psum, pmax);
  k_select<<<1, THREADS, 0, st>>>(out, s.probs, s.vocab, psum, p.top_p, p.top_k,
                                  p.min_p, p.seed, step, greedy ? 1 : 0);
}

}  // namespace qwen
