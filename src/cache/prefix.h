// Prefix caching for a hybrid GatedDeltaNet + attention model.
//
// THE PROBLEM. A KV cache can be truncated to a prefix: position p's keys and
// values do not depend on anything after p, so reusing [0, p) is just a matter
// of not overwriting it. A GDN layer's recurrent state cannot. It is updated in
// place, S_t is a function of S_{t-1}, and there is no way to run it backwards
// to recover the state at an earlier position. Everyone serving these models
// hits this -- vLLM tracks it as an open issue, SGLang built MambaRadixCache,
// LMCache treats the state as an opaque page.
//
// So a prefix is reusable only at positions where the recurrent state was
// SNAPSHOTTED. The state is "all or nothing": either you have it at exactly
// position p or you do not.
//
// WHERE TO SNAPSHOT. Marconi (arXiv 2411.19379) measured that at 32-token
// blocks, 25% of KV blocks get reused by a later request but only 0.4% of SSM
// states do -- a 65x gap. Snapshotting periodically is therefore almost pure
// waste. Their answer is to admit at most a couple of states per sequence: at
// points where the radix tree branches, and at the last decoded token. The
// second one is the important one here, because "the last decoded token" is
// exactly where a chat turn ends, and turn N+1's prompt is turn N's context
// plus new user text. On SWE-Bench agent traces that policy is worth 34.4x on
// hit rate and 36-71% of P95 TTFT.
//
// WHERE WE DIFFER. Marconi keeps snapshots in GPU memory because it serves many
// sequences concurrently and has to. We serve one, so snapshots live in PINNED
// HOST memory: 150 MiB moves over PCIe in about 13 ms, against a prefill that
// costs seconds. That buys slots for the price of host RAM and leaves the whole
// device budget to the KV cache, so prefix caching and a maximal context are
// not in competition.
//
// EXACTNESS, in three parts, because they are not equally strong and gate_prefix
// tests them separately:
//
//   1. Store and restore is a bit-for-bit fp32 round trip. Exact, always.
//   2. Resuming from a snapshot that sits on a PREFILL CHUNK BOUNDARY is exact:
//      the chunks that remain have identical shapes to the ones a cold run would
//      use, so the arithmetic is identical. Measured: 0 of 248320 logit bits
//      differ. This is what vLLM calls mamba-cache-mode "align".
//   3. Resuming from a snapshot taken AFTER GENERATION is not exact and cannot
//      be. That state was produced by D single-token decode steps; a cold run
//      reaches the same position with a large chunked prefill. Different
//      kernels, different summation orders. Measured: top-1 agrees and
//      KL(cold||warm) = 3.75e-04, which is smaller than the 7e-4 this model's
//      INT4 weights already cost against the bf16 reference -- so the cache is
//      well inside the numerical envelope we already accepted, but it is not
//      bitwise, and saying otherwise would be false.
//
// Mode 3 is the only one that can reuse generated tokens, and therefore the only
// one that reaches the directive's 20x bar (measured 46.4x). Mode 2 is available
// for callers that need run-to-run bitwise reproducibility more than they need
// the last few hundred tokens of reuse.
#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include "../model/model.h"

namespace qwen {

struct PrefixSlot {
  std::vector<int32_t> tokens;   // the prefix this state represents; size() == pos
  int      pos = 0;
  float*   h_state = nullptr;    // pinned host, [gdn_state_elems]
  float*   h_conv = nullptr;     // pinned host, [gdn_conv_state_elems]
  uint16_t* h_logits = nullptr;  // pinned host, [vocab] -- see note below
  uint64_t stamp = 0;            // for LRU
  bool     valid = false;
};

struct PrefixCache {
  std::vector<PrefixSlot> slots;

  // What the DEVICE KV arena currently holds. A snapshot at position p is only
  // usable if the KV for [0, p) is still the KV for those exact tokens, and the
  // arena is overwritten whenever a request diverges from it. Tracking the
  // arena's contents is what makes that check cheap and correct.
  std::vector<int32_t> kv_tokens;

  int64_t  state_elems = 0, conv_elems = 0;
  int      vocab = 0;
  uint64_t clock = 0;
  size_t   host_bytes = 0;

  // stats
  uint64_t hits = 0, misses = 0, reused_tokens = 0, prefilled_tokens = 0;
  double   restore_ms = 0, store_ms = 0;
};

void prefix_alloc(PrefixCache& c, const Model& m, int n_slots);
void prefix_free(PrefixCache& c);

// Drop every snapshot and forget the KV arena's contents.
void prefix_reset(PrefixCache& c);

// Longest prefix of `tokens` that can be resumed. Returns the number of tokens
// that need not be prefilled, and writes the winning slot to *slot (-1 on miss).
//
// This CAN return tokens.size(): the snapshot carries the logits for its last
// position, so a prompt that is exactly a cached prefix needs no prefill at all.
// That is the resend case -- same conversation, different sampling params.
int prefix_lookup(PrefixCache& c, const std::vector<int32_t>& tokens, int* slot);

// Copy a slot's recurrent state back onto the device and set m.ctx_len.
void prefix_restore(PrefixCache& c, Model& m, int slot);

// Zero the recurrent state, for a cold start.
void prefix_cold(Model& m);

// Snapshot the model's current recurrent state as representing tokens[0, pos).
// Call at the end of prefill and at the end of generation; the second is the one
// that makes the next turn of a conversation nearly free.
void prefix_store(PrefixCache& c, Model& m, const std::vector<int32_t>& tokens, int pos);

// Record that the device KV arena now holds exactly these tokens.
void prefix_set_kv(PrefixCache& c, const std::vector<int32_t>& tokens);

}  // namespace qwen
