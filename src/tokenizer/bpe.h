// Qwen (Qwen2Tokenizer) byte-level BPE, pure C++.
//
// Pipeline, matching HF exactly:
//   raw text
//     -> split on added tokens (all 33 have normalized=false, so they are
//        matched against RAW text, before normalization)
//     -> per segment: NFC  ->  pretokenizer regex  ->  byte-level map  ->  BPE
//
// The pretokenizer pattern is
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   |\p{N}
//   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
// It contains a negative lookahead, so HF runs it under fancy-regex, which is a
// backtracking engine with leftmost-FIRST alternation. match_at() below
// reproduces that ordering and the two places backtracking actually changes the
// result; see the comments there. \s is Unicode White_Space, not isspace().
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace qwen {

class Tokenizer {
 public:
  // Loads vocab, merges and added tokens from a HF tokenizer.json.
  // Throws std::runtime_error on a malformed or unexpected file rather than
  // silently degrading: a tokenizer that is subtly wrong is worse than one that
  // refuses to start.
  void load(const std::string& tokenizer_json_path);

  std::vector<int32_t> encode(std::string_view text, bool allow_special = true) const;
  std::string decode(const std::vector<int32_t>& ids, bool skip_special = false) const;

  int32_t token_to_id(const std::string& tok) const;
  const std::string& id_to_token(int32_t id) const;

  size_t base_vocab_size() const { return id_to_tok_.size(); }
  int32_t eos_id() const { return eos_id_; }
  int32_t pad_id() const { return pad_id_; }
  bool is_special(int32_t id) const {
    auto it = special_ids_.find(id);
    return it != special_ids_.end();
  }

  // Exposed for tests: the pretokenizer split of one already-NFC'd segment.
  std::vector<std::string> pretokenize(std::string_view seg) const;

 private:
  std::unordered_map<std::string, int32_t> tok_to_id_;
  std::vector<std::string>                 id_to_tok_;
  // merge rank keyed by (lhs_id << 32 | rhs_id); value is (rank, merged_id)
  std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges_;

  struct Added { std::string content; int32_t id; bool special; };
  std::vector<Added> added_;                       // sorted by content length desc
  std::unordered_map<int32_t, bool> special_ids_;  // id -> special?

  int32_t eos_id_ = -1, pad_id_ = -1;

  // byte-level maps
  std::string byte_to_uni_[256];                   // byte -> UTF-8 of its codepoint
  std::unordered_map<uint32_t, uint8_t> uni_to_byte_;

  void bpe_word(const std::string& bytelevel, std::vector<int32_t>& out) const;
};

}  // namespace qwen
