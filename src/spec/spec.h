// Speculative decoding: block verification, acceptance, and state rollback.
//
// The acceptance rule is NOT invented here. Greedy is the longest matching
// prefix of exact argmax equality, stopping at the first mismatch, plus the
// target's own correction token -- exactly what z-lab/dflash's model.py does.
// Sampling uses Leviathan/Chen rejection sampling. There is no `tau`: the
// official DFlash repo has no acceptance threshold of any kind, and the one
// that exists in the wild is a third-party lossy addition.
//
// ROLLBACK is the hard part on a hybrid model. KV rollback is a length pointer.
// The GatedDeltaNet recurrent state is not: by the time acceptance is known the
// state has already been advanced through the whole block.
//
// The design here is the one the official MLX reference uses (model_mlx.py
// _GDNStateCapture), and the directive's option (A): snapshot the seed state,
// capture the per-layer recurrence INPUTS during the verify forward, and on
// partial acceptance restore the seed and replay the recurrence over only the
// accepted prefix. The short-conv state is SLICED from the captured input
// window rather than replayed. Replaying <= 8 tokens through 48 layers is cheap
// because the projections -- the expensive part -- are not redone.
#pragma once
#include <cstdint>
#include <vector>
#include <cuda_bf16.h>
#include "../model/model.h"

namespace qwen {

struct SpecState {
  // seed snapshots, taken before each verify block
  float* gdn_state_bak = nullptr;     // [gdn_state_elems]
  float* gdn_conv_bak  = nullptr;     // [gdn_conv_state_elems]
  // per-GDN-layer recurrence inputs captured during the verify forward
  __nv_bfloat16* pre_conv = nullptr;  // [gdn_layers][block][conv_dim]
  __nv_bfloat16* post_conv = nullptr; // [gdn_layers][block][conv_dim]
  float* g = nullptr;                 // [gdn_layers][block][v_heads]
  float* beta = nullptr;              // [gdn_layers][block][v_heads]
  int    block = 0;
  bool   capturing = false;
  std::vector<void*> owned;
};

void spec_alloc(SpecState& s, const Model& m, int max_block);
void spec_free(SpecState& s);

// Snapshot the seed state and arm capture for the next forward.
void spec_begin_block(SpecState& s, Model& m);
// Restore the seed and replay the recurrence over `n` accepted positions.
void spec_rollback(SpecState& s, Model& m, int n);

// Runs the target over `ids[0..T)` at `position`, producing logits for EVERY
// position rather than only the last.
void model_verify_block(Model& m, const int32_t* ids, int T, int position,
                        __nv_bfloat16* logits_out);

// Greedy acceptance: the longest prefix where the target's argmax at position i
// equals the drafted token i+1, then the target's own token at the first
// mismatch. Returns the number of DRAFTED tokens accepted; the caller always
// also commits the correction.
int accept_greedy(const std::vector<int32_t>& drafted,
                  const std::vector<int32_t>& target_argmax,
                  int32_t& correction);

// ---------------------------------------------------------------- drafters
// Suffix / n-gram drafter. Zero weight reads: it proposes the continuation of
// the most recent earlier occurrence of the current suffix. Useless on prose,
// strong on code and on agentic transcripts where the model quotes its own
// context back. Kept because it costs nothing and is a floor under any
// learned drafter.
class SuffixDrafter {
 public:
  void reset() { seq_.clear(); }
  void push(int32_t t) { seq_.push_back(t); }
  const std::vector<int32_t>& sequence() const { return seq_; }
  // Returns up to k proposed tokens; empty if no match.
  std::vector<int32_t> propose(int k, int min_match = 3, int max_match = 12) const;
 private:
  std::vector<int32_t> seq_;
};

// One speculative round.
//
// Positions: committed tokens run to `pos-1` and m.logits holds the
// distribution for position `pos`. The block is [t0, d1..dk] at positions
// pos..pos+k, where t0 = argmax(m.logits) is FREE -- it was already computed.
// logits[i] of the verify pass predicts position pos+i+1, so accepting `a`
// drafted tokens commits t0 plus d1..da, and the target's own token at slot a
// becomes the next round's free token.
struct RoundResult {
  std::vector<int32_t> committed;   // tokens to emit this round
  int accepted = 0;                 // drafted tokens accepted
  int drafted = 0;
  bool hit_stop = false;
};

struct SpecStats {
  uint64_t rounds = 0, drafted = 0, accepted = 0, committed = 0;
  uint64_t fallback_tokens = 0;   // steps that skipped speculation as unprofitable
  std::vector<uint64_t> per_position;   // acceptance histogram by draft slot
  double mean_acceptance() const {
    return rounds ? double(committed) / double(rounds) : 0.0;
  }
};

// Greedy generation with speculation. Returns the emitted token ids.
std::vector<int32_t> spec_generate(Model& m, SpecState& s,
                                   const std::vector<int32_t>& prompt,
                                   int max_new, int max_k, SpecStats& stats);

}  // namespace qwen
