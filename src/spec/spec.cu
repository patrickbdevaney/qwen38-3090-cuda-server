#include "spec.h"
#include "../kernels/gdn.cuh"
#include "../kernels/elementwise.cuh"
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <cstring>
#include <cuda_runtime.h>

namespace qwen {
namespace {
#define CKS(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); abort(); } } while(0)

void* salloc(SpecState& s, size_t n) {
  void* p = nullptr; CKS(cudaMalloc(&p, n)); s.owned.push_back(p); return p;
}

// New conv state after committing `n` block positions: the last K-1 inputs of
// [old_state | block_input[0..n)]. A slice, not a recompute.
__global__ void k_conv_state_slice(float* __restrict__ state,
                                   const float* __restrict__ old_state,
                                   const __nv_bfloat16* __restrict__ pre,
                                   int conv_dim, int K, int n, int block) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(conv_dim) * (K - 1)) return;
  const int c = int(i / (K - 1)), j = int(i % (K - 1));
  const int t = n - (K - 1) + j;                 // position relative to block start
  if (t >= 0) state[i] = __bfloat162float(pre[size_t(t) * conv_dim + c]);
  else        state[i] = old_state[size_t(c) * (K - 1) + (K - 1 + t)];
}
}  // namespace

void spec_alloc(SpecState& s, const Model& m, int max_block) {
  s.block = max_block;
  const auto& S = m.shape;
  const int NG = S.num_gdn_layers, CD = int(m.gdn.conv_dim()), NV = S.linear_num_value_heads;
  s.gdn_state_bak = static_cast<float*>(salloc(s, size_t(S.gdn_state_elems()) * 4));
  s.gdn_conv_bak  = static_cast<float*>(salloc(s, size_t(S.gdn_conv_state_elems()) * 4));
  s.pre_conv  = static_cast<__nv_bfloat16*>(salloc(s, size_t(NG) * max_block * CD * 2));
  s.post_conv = static_cast<__nv_bfloat16*>(salloc(s, size_t(NG) * max_block * CD * 2));
  s.g    = static_cast<float*>(salloc(s, size_t(NG) * max_block * NV * 4));
  s.beta = static_cast<float*>(salloc(s, size_t(NG) * max_block * NV * 4));
}

void spec_free(SpecState& s) {
  for (void* p : s.owned) cudaFree(p);
  s.owned.clear();
  s = SpecState{};
}

void spec_begin_block(SpecState& s, Model& m) {
  const auto& S = m.shape;
  CKS(cudaMemcpyAsync(s.gdn_state_bak, m.gdn_state, size_t(S.gdn_state_elems()) * 4,
                      cudaMemcpyDeviceToDevice, m.stream));
  CKS(cudaMemcpyAsync(s.gdn_conv_bak, m.gdn_conv, size_t(S.gdn_conv_state_elems()) * 4,
                      cudaMemcpyDeviceToDevice, m.stream));
  s.capturing = true;
  m.spec = &s;
}

void spec_rollback(SpecState& s, Model& m, int n) {
  const auto& S = m.shape;
  const int NG = S.num_gdn_layers, CD = int(m.gdn.conv_dim()), NV = S.linear_num_value_heads;
  // restore the seed
  CKS(cudaMemcpyAsync(m.gdn_state, s.gdn_state_bak, size_t(S.gdn_state_elems()) * 4,
                      cudaMemcpyDeviceToDevice, m.stream));
  if (n <= 0) {
    CKS(cudaMemcpyAsync(m.gdn_conv, s.gdn_conv_bak, size_t(S.gdn_conv_state_elems()) * 4,
                        cudaMemcpyDeviceToDevice, m.stream));
    return;
  }
  const size_t st_per = size_t(NV) * m.gdn.head_k * m.gdn.head_v;
  for (int gi = 0; gi < NG; ++gi) {
    // replay the recurrence over the accepted prefix only; the projections and
    // the conv are NOT redone, which is what makes this cheap
    gdn_scan(m.gdn_core,
             m.gdn_state + size_t(gi) * st_per,
             s.post_conv + size_t(gi) * s.block * CD,
             s.g + size_t(gi) * s.block * NV,
             s.beta + size_t(gi) * s.block * NV,
             m.gdn, n, m.stream);
    const size_t cs = size_t(CD) * (m.gdn.conv_k - 1);
    k_conv_state_slice<<<(cs + 255) / 256, 256, 0, m.stream>>>(
        m.gdn_conv + size_t(gi) * cs, s.gdn_conv_bak + size_t(gi) * cs,
        s.pre_conv + size_t(gi) * s.block * CD, CD, m.gdn.conv_k, n, s.block);
  }
}

int accept_greedy(const std::vector<int32_t>& drafted,
                  const std::vector<int32_t>& target_argmax,
                  int32_t& correction) {
  size_t a = 0;
  while (a < drafted.size() && a < target_argmax.size() && target_argmax[a] == drafted[a]) ++a;
  correction = (a < target_argmax.size()) ? target_argmax[a] : target_argmax.back();
  return int(a);
}

std::vector<int32_t> SuffixDrafter::propose(int k, int min_match, int max_match) const {
  const int n = int(seq_.size());
  if (n < min_match + 1) return {};
  for (int L = std::min(max_match, n - 1); L >= min_match; --L) {
    // most recent earlier occurrence wins. The search must stop BEFORE the
    // current suffix itself, or every query trivially matches itself and the
    // drafter silently never fires -- a bug laguna's suffix.h documents.
    for (int i = n - L - 1; i >= 0; --i) {
      bool ok = true;
      for (int j = 0; j < L; ++j)
        if (seq_[i + j] != seq_[n - L + j]) { ok = false; break; }
      if (!ok) continue;
      std::vector<int32_t> out;
      for (int j = 0; j < k && i + L + j < n; ++j) out.push_back(seq_[i + L + j]);
      if (!out.empty()) return out;
    }
  }
  return {};
}

void model_verify_block(Model& m, const int32_t* ids, int T, int position,
                        __nv_bfloat16* logits_out) {
  model_forward_all_logits(m, ids, T, position, logits_out);
}

namespace {
int32_t argmax_host(const std::vector<uint16_t>& v, size_t off, size_t n) {
  auto f = [](uint16_t h) { uint32_t u = uint32_t(h) << 16; float x; memcpy(&x, &u, 4); return x; };
  float best = -1e30f; int32_t bi = 0;
  for (size_t i = 0; i < n; ++i) {
    const float x = f(v[off + i]);
    if (x > best) { best = x; bi = int32_t(i); }   // strict >: lowest index wins ties
  }
  return bi;
}
}  // namespace

std::vector<int32_t> spec_generate(Model& m, SpecState& s,
                                   const std::vector<int32_t>& prompt,
                                   int max_new, int max_k, SpecStats& stats) {
  const int V = m.shape.vocab_size;
  std::vector<int32_t> out;
  SuffixDrafter drafter;
  for (int32_t t : prompt) drafter.push(t);

  __nv_bfloat16* lg = nullptr;
  CKS(cudaMalloc(&lg, size_t(max_k + 1) * V * 2));
  std::vector<uint16_t> host(size_t(max_k + 1) * V);

  // A verification block of T rows costs more than one decode step, so
  // speculation only pays when the committed tokens per round exceed that cost.
  // bench_block measures the ratio directly on this card and this checkpoint:
  // T=2 -> 1.92x, T=4 -> 2.04x, T=8 -> 2.11x, T=16 -> 2.23x, i.e. very nearly
  // 1.876 + 0.0221*T. A drafter that lands 2.0 tokens per round at k=7 is a NET
  // LOSS, which is exactly what the suffix drafter does on non-repetitive text,
  // so track the realised gain and fall back to plain decode when it stops
  // paying. Cooldown doubles on repeated failure and resets on success, so a
  // prompt that turns repetitive halfway through still gets speculation back.
  const double cost_a = getenv("QWEN_SPEC_COST_A") ? atof(getenv("QWEN_SPEC_COST_A")) : 1.876;
  const double cost_b = getenv("QWEN_SPEC_COST_B") ? atof(getenv("QWEN_SPEC_COST_B")) : 0.0221;
  double ewma = 0.0; bool have_ewma = false;
  int cooldown = 0, penalty = 8;

  int pos = int(prompt.size());
  // chunked prefill
  {
    int p = 0;
    while (p < int(prompt.size())) {
      const int n = std::min(m.max_batch, int(prompt.size()) - p);
      model_prefill(m, prompt.data() + p, n, p);
      p += n;
    }
  }
  stats.per_position.assign(max_k + 1, 0);

  while (int(out.size()) < max_new) {
    // the free token: already computed by the previous step
    std::vector<uint16_t> l1(V);
    CKS(cudaMemcpy(l1.data(), m.logits, size_t(V) * 2, cudaMemcpyDeviceToHost));
    const int32_t t0 = argmax_host(l1, 0, V);
    out.push_back(t0);
    drafter.push(t0);
    if (m.shape.is_stop_token(t0) || int(out.size()) >= max_new) break;

    std::vector<int32_t> drafted = drafter.propose(max_k);
    if (cooldown > 0) { --cooldown; ++stats.fallback_tokens; drafted.clear(); }
    if (drafted.empty()) {
      model_decode(m, t0, pos);
      ++pos;
      continue;
    }
    const int k = int(drafted.size());
    std::vector<int32_t> block;
    block.push_back(t0);
    for (int32_t d : drafted) block.push_back(d);

    spec_begin_block(s, m);
    model_verify_block(m, block.data(), k + 1, pos, lg);
    m.spec = nullptr; s.capturing = false;
    CKS(cudaMemcpy(host.data(), lg, size_t(k + 1) * V * 2, cudaMemcpyDeviceToHost));

    std::vector<int32_t> targ(k + 1);
    for (int i = 0; i <= k; ++i) targ[i] = argmax_host(host, size_t(i) * V, V);

    int32_t correction = 0;
    const int a = accept_greedy(drafted, targ, correction);

    ++stats.rounds;
    stats.drafted += k;
    stats.accepted += a;
    stats.committed += a + 1;                 // t0 plus the accepted drafts
    if (a < int(stats.per_position.size())) stats.per_position[a]++;

    const double gain = double(a + 1);
    ewma = have_ewma ? 0.8 * ewma + 0.2 * gain : gain;
    have_ewma = true;
    if (ewma < cost_a + cost_b * double(k + 1)) {
      cooldown = penalty;
      penalty = std::min(penalty * 2, 256);
    } else {
      penalty = 8;
    }

    for (int i = 0; i < a; ++i) { out.push_back(drafted[i]); drafter.push(drafted[i]); }

    // Roll the recurrent state back to the a+1 positions actually committed.
    // KV needs nothing: the next write simply starts at pos + a + 1 and
    // overwrites the rejected tail.
    if (a + 1 < k + 1) spec_rollback(s, m, a + 1);
    pos += a + 1;

    // the correction's distribution becomes the next round's free token
    CKS(cudaMemcpy(m.logits, lg + size_t(a) * V, size_t(V) * 2, cudaMemcpyDeviceToDevice));
    if (m.shape.is_stop_token(correction)) { out.push_back(correction); break; }
  }
  cudaFree(lg);
  return out;
}

}  // namespace qwen
