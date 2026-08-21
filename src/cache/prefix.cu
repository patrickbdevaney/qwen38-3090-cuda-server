#include "prefix.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>

namespace qwen {
namespace {

#define CKP(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); abort(); } } while(0)

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

// How many leading tokens two sequences share.
size_t common_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

}  // namespace

void prefix_alloc(PrefixCache& c, const Model& m, int n_slots) {
  c.state_elems = m.shape.gdn_state_elems();
  c.conv_elems = m.shape.gdn_conv_state_elems();
  c.vocab = m.shape.vocab_size;
  c.slots.resize(std::max(0, n_slots));
  const size_t sb = size_t(c.state_elems) * 4, cb = size_t(c.conv_elems) * 4;
  for (auto& s : c.slots) {
    // Pinned, so the copies are real DMA and can overlap. 150 MiB a slot is
    // nothing against 60 GB of host RAM, and it costs zero device memory.
    CKP(cudaHostAlloc(reinterpret_cast<void**>(&s.h_state), sb, cudaHostAllocDefault));
    CKP(cudaHostAlloc(reinterpret_cast<void**>(&s.h_conv), cb, cudaHostAllocDefault));
    // The logits for the snapshot's last position. Without them a restore has
    // no "current" distribution and the caller must re-run at least one token;
    // with them the state is genuinely complete. 0.5 MB.
    CKP(cudaHostAlloc(reinterpret_cast<void**>(&s.h_logits), size_t(c.vocab) * 2,
                      cudaHostAllocDefault));
    c.host_bytes += sb + cb + size_t(c.vocab) * 2;
  }
}

void prefix_free(PrefixCache& c) {
  for (auto& s : c.slots) {
    if (s.h_state) cudaFreeHost(s.h_state);
    if (s.h_conv) cudaFreeHost(s.h_conv);
    if (s.h_logits) cudaFreeHost(s.h_logits);
  }
  c.slots.clear();
  c.kv_tokens.clear();
  c.host_bytes = 0;
}

void prefix_reset(PrefixCache& c) {
  for (auto& s : c.slots) { s.valid = false; s.pos = 0; s.tokens.clear(); }
  c.kv_tokens.clear();
}

void prefix_cold(Model& m) {
  CKP(cudaMemset(m.gdn_state, 0, size_t(m.shape.gdn_state_elems()) * 4));
  CKP(cudaMemset(m.gdn_conv, 0, size_t(m.shape.gdn_conv_state_elems()) * 4));
  m.ctx_len = 0;
}

int prefix_lookup(PrefixCache& c, const std::vector<int32_t>& tokens, int* slot) {
  *slot = -1;
  if (tokens.empty()) return 0;

  // The device KV arena is the hard limit: a snapshot's state is only useful if
  // the keys and values for those positions are still the right ones.
  const size_t kv_ok = common_prefix(tokens, c.kv_tokens);

  int best = -1, best_pos = 0;
  for (size_t i = 0; i < c.slots.size(); ++i) {
    const PrefixSlot& s = c.slots[i];
    if (!s.valid || s.pos <= best_pos) continue;
    if (size_t(s.pos) > kv_ok) continue;               // KV for that span is gone
    if (common_prefix(s.tokens, tokens) < size_t(s.pos)) continue;
    best = int(i); best_pos = s.pos;
  }
  if (best < 0) { ++c.misses; return 0; }
  ++c.hits;
  *slot = best;
  c.reused_tokens += uint64_t(best_pos);
  return best_pos;
}

void prefix_restore(PrefixCache& c, Model& m, int slot) {
  PrefixSlot& s = c.slots[slot];
  const double t0 = now_ms();
  CKP(cudaMemcpy(m.gdn_state, s.h_state, size_t(c.state_elems) * 4, cudaMemcpyHostToDevice));
  CKP(cudaMemcpy(m.gdn_conv, s.h_conv, size_t(c.conv_elems) * 4, cudaMemcpyHostToDevice));
  CKP(cudaMemcpy(m.logits, s.h_logits, size_t(c.vocab) * 2, cudaMemcpyHostToDevice));
  m.ctx_len = s.pos;
  s.stamp = ++c.clock;
  c.restore_ms += now_ms() - t0;
}

void prefix_store(PrefixCache& c, Model& m, const std::vector<int32_t>& tokens, int pos) {
  if (c.slots.empty() || pos <= 0 || size_t(pos) > tokens.size()) return;

  // Refresh in place if we already hold this exact prefix, so re-running the
  // same conversation does not consume a slot per turn.
  int target = -1;
  for (size_t i = 0; i < c.slots.size(); ++i) {
    PrefixSlot& s = c.slots[i];
    if (s.valid && s.pos == pos && common_prefix(s.tokens, tokens) >= size_t(pos)) {
      target = int(i);
      break;
    }
  }
  if (target < 0) {
    // Prefer a free slot, else evict least recently used. Plain LRU: Marconi
    // needs a FLOP-efficiency score because its snapshots compete with the KV
    // cache for device memory, and ours do not compete with anything.
    uint64_t oldest = UINT64_MAX;
    for (size_t i = 0; i < c.slots.size(); ++i) {
      if (!c.slots[i].valid) { target = int(i); break; }
      if (c.slots[i].stamp < oldest) { oldest = c.slots[i].stamp; target = int(i); }
    }
  }

  PrefixSlot& s = c.slots[target];
  const double t0 = now_ms();
  CKP(cudaMemcpy(s.h_state, m.gdn_state, size_t(c.state_elems) * 4, cudaMemcpyDeviceToHost));
  CKP(cudaMemcpy(s.h_conv, m.gdn_conv, size_t(c.conv_elems) * 4, cudaMemcpyDeviceToHost));
  CKP(cudaMemcpy(s.h_logits, m.logits, size_t(c.vocab) * 2, cudaMemcpyDeviceToHost));
  s.tokens.assign(tokens.begin(), tokens.begin() + pos);
  s.pos = pos;
  s.valid = true;
  s.stamp = ++c.clock;
  c.store_ms += now_ms() - t0;
}

void prefix_set_kv(PrefixCache& c, const std::vector<int32_t>& tokens) {
  c.kv_tokens = tokens;
}

}  // namespace qwen
