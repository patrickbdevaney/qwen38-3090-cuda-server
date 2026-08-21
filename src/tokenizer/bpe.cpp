#include "bpe.h"
#include "unicode.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include "../../third_party/json.hpp"

using json = nlohmann::json;

namespace qwen {
namespace {

// GPT-2 byte-level alphabet: an invertible byte <-> codepoint map that keeps
// every byte printable and never produces whitespace, so the pretokenizer's
// output can be BPE'd as text.
std::vector<uint32_t> byte_alphabet() {
  std::vector<uint32_t> b2u(256, 0);
  std::vector<bool> assigned(256, false);
  auto take = [&](int lo, int hi) {
    for (int b = lo; b <= hi; ++b) { b2u[b] = static_cast<uint32_t>(b); assigned[b] = true; }
  };
  take('!', '~'); take(0xA1, 0xAC); take(0xAE, 0xFF);
  uint32_t n = 0;
  for (int b = 0; b < 256; ++b)
    if (!assigned[b]) { b2u[b] = 256 + n; ++n; }
  return b2u;
}

inline bool is_cr_lf(uint32_t cp) { return cp == '\r' || cp == '\n'; }

// [^\s\p{L}\p{M}\p{N}]
inline bool is_other(uint32_t cp) {
  return !is_space(cp) && !is_letter(cp) && !is_mark(cp) && !is_number(cp);
}

// One case-insensitive ASCII compare, for the contraction alternative.
inline bool ieq(uint32_t cp, char c) {
  uint32_t l = (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp;
  return l == static_cast<uint32_t>(c);
}

// Returns the length in codepoints of the match at `i`, or 0 if none.
// Alternatives are tried in source order: fancy-regex is leftmost-first.
size_t match_at(const std::vector<uint32_t>& c, size_t i) {
  const size_t n = c.size();

  // 1. (?i:'s|'t|'re|'ve|'m|'ll|'d)
  if (c[i] == '\'') {
    if (i + 1 < n) {
      const uint32_t a = c[i + 1];
      if (ieq(a, 's') || ieq(a, 't') || ieq(a, 'm') || ieq(a, 'd')) return 2;
      if (i + 2 < n) {
        const uint32_t b = c[i + 2];
        if ((ieq(a, 'r') && ieq(b, 'e')) ||
            (ieq(a, 'v') && ieq(b, 'e')) ||
            (ieq(a, 'l') && ieq(b, 'l'))) return 3;
      }
    }
  }

  // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
  //    The optional class CAN match a Mark (marks are neither L nor N), and
  //    [\p{L}\p{M}]+ can also start with a Mark. So when the optional char is
  //    consumed and no letter/mark follows, the engine backtracks to not
  //    consuming it and retries -- which succeeds exactly when c[i] is a Mark.
  {
    auto letters_from = [&](size_t j) {
      size_t k = j;
      while (k < n && (is_letter(c[k]) || is_mark(c[k]))) ++k;
      return k;
    };
    const bool opt_ok = !is_cr_lf(c[i]) && !is_letter(c[i]) && !is_number(c[i]);
    if (opt_ok) {
      const size_t k = letters_from(i + 1);
      if (k > i + 1) return k - i;                 // optional consumed
    }
    const size_t k = letters_from(i);              // optional not consumed
    if (k > i) return k - i;
  }

  // 3. \p{N}   (single codepoint, unlike GPT-2's \p{N}{1,3})
  if (is_number(c[i])) return 1;

  // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
  //    Backtracking on the optional space cannot help: if the space is consumed
  //    and the class then fails, retrying without it fails too, because a space
  //    is whitespace and so is excluded by the class.
  {
    const size_t j = (c[i] == ' ') ? i + 1 : i;
    size_t k = j;
    while (k < n && is_other(c[k])) ++k;
    if (k > j) {
      while (k < n && is_cr_lf(c[k])) ++k;
      return k - i;
    }
  }

  // 5. \s*[\r\n]+
  //    Greedy \s* takes the whole whitespace run, then [\r\n]+ must match, so
  //    the engine backtracks until the match ENDS on a CR/LF. Net effect: match
  //    the run up to and including its last CR/LF, if it has one.
  {
    size_t w = i;
    while (w < n && is_space(c[w])) ++w;
    size_t q = w;
    while (q > i && !is_cr_lf(c[q - 1])) --q;      // strip trailing non-newline ws
    if (q > i) return q - i;
  }

  // 6. \s+(?!\S)
  //    \s+ is greedy and the lookahead forbids a following non-space. If the
  //    run is followed by a non-space, the engine gives back exactly one
  //    character so the lookahead sees whitespace and succeeds. If that would
  //    leave the match empty, the alternative fails and we fall through to 7.
  {
    size_t w = i;
    while (w < n && is_space(c[w])) ++w;
    if (w > i) {
      if (w == n) return w - i;                    // run reaches end of input
      if (w - 1 > i) return (w - 1) - i;           // give back one character
    }
  }

  // 7. \s+
  {
    size_t w = i;
    while (w < n && is_space(c[w])) ++w;
    if (w > i) return w - i;
  }

  return 0;
}

}  // namespace

// ---------------------------------------------------------------- load
void Tokenizer::load(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("tokenizer: cannot open " + path);
  json j;
  f >> j;

  const auto& model = j.at("model");
  if (model.at("type").get<std::string>() != "BPE")
    throw std::runtime_error("tokenizer: expected model.type == BPE");

  // The C++ side assumes these; if a future checkpoint changes them the
  // pipeline above is wrong and we must fail loudly rather than drift.
  auto expect_false = [&](const char* k) {
    if (model.contains(k) && model.at(k).is_boolean() && model.at(k).get<bool>())
      throw std::runtime_error(std::string("tokenizer: unsupported model.") + k);
  };
  expect_false("fuse_unk");
  expect_false("byte_fallback");
  expect_false("ignore_merges");
  if (j.contains("normalizer") && !j.at("normalizer").is_null()) {
    const auto t = j.at("normalizer").value("type", "");
    if (t != "NFC") throw std::runtime_error("tokenizer: expected NFC normalizer, got " + t);
  }

  const auto& vocab = model.at("vocab");
  id_to_tok_.assign(vocab.size(), std::string());
  tok_to_id_.reserve(vocab.size() * 2);
  for (auto it = vocab.begin(); it != vocab.end(); ++it) {
    const int32_t id = it.value().get<int32_t>();
    if (id < 0) throw std::runtime_error("tokenizer: negative vocab id");
    if (static_cast<size_t>(id) >= id_to_tok_.size()) id_to_tok_.resize(id + 1);
    id_to_tok_[id] = it.key();
    tok_to_id_.emplace(it.key(), id);
  }

  // merges: either "A B" strings or ["A","B"] pairs depending on the version
  const auto& merges = model.at("merges");
  merges_.reserve(merges.size() * 2);
  int32_t rank = 0;
  for (const auto& m : merges) {
    std::string a, b;
    if (m.is_string()) {
      const std::string s = m.get<std::string>();
      const size_t sp = s.find(' ');
      if (sp == std::string::npos) throw std::runtime_error("tokenizer: bad merge " + s);
      a = s.substr(0, sp); b = s.substr(sp + 1);
    } else if (m.is_array() && m.size() == 2) {
      a = m[0].get<std::string>(); b = m[1].get<std::string>();
    } else {
      throw std::runtime_error("tokenizer: unrecognised merges entry");
    }
    const auto ia = tok_to_id_.find(a), ib = tok_to_id_.find(b),
               ic = tok_to_id_.find(a + b);
    ++rank;
    if (ia == tok_to_id_.end() || ib == tok_to_id_.end() || ic == tok_to_id_.end())
      continue;   // a merge whose product is not in the vocab can never fire
    const uint64_t key = (static_cast<uint64_t>(ia->second) << 32) |
                          static_cast<uint32_t>(ib->second);
    merges_.emplace(key, std::make_pair(rank - 1, ic->second));
  }

  if (j.contains("added_tokens")) {
    for (const auto& a : j.at("added_tokens")) {
      Added ad{a.at("content").get<std::string>(), a.at("id").get<int32_t>(),
               a.value("special", false)};
      // Only the simple flag combination is implemented; anything else would
      // change matching semantics silently.
      if (a.value("lstrip", false) || a.value("rstrip", false) ||
          a.value("single_word", false) || a.value("normalized", false))
        throw std::runtime_error("tokenizer: unsupported added_token flags on " + ad.content);
      special_ids_[ad.id] = ad.special;
      if (static_cast<size_t>(ad.id) >= id_to_tok_.size()) id_to_tok_.resize(ad.id + 1);
      id_to_tok_[ad.id] = ad.content;
      tok_to_id_.emplace(ad.content, ad.id);
      added_.push_back(std::move(ad));
    }
    // Longest content first, so "<|im_start|>" wins over any shorter prefix.
    std::sort(added_.begin(), added_.end(),
              [](const Added& x, const Added& y) {
                if (x.content.size() != y.content.size())
                  return x.content.size() > y.content.size();
                return x.content < y.content;
              });
  }

  auto find = [&](const char* s) {
    auto it = tok_to_id_.find(s);
    return it == tok_to_id_.end() ? -1 : it->second;
  };
  eos_id_ = find("<|im_end|>");
  pad_id_ = find("<|endoftext|>");

  const auto b2u = byte_alphabet();
  for (int b = 0; b < 256; ++b) {
    std::string s;
    utf8_append(s, b2u[b]);
    byte_to_uni_[b] = s;
    uni_to_byte_[b2u[b]] = static_cast<uint8_t>(b);
  }
}

int32_t Tokenizer::token_to_id(const std::string& t) const {
  auto it = tok_to_id_.find(t);
  return it == tok_to_id_.end() ? -1 : it->second;
}

const std::string& Tokenizer::id_to_token(int32_t id) const {
  static const std::string kEmpty;
  if (id < 0 || static_cast<size_t>(id) >= id_to_tok_.size()) return kEmpty;
  return id_to_tok_[id];
}

// ---------------------------------------------------------------- pretokenize
std::vector<std::string> Tokenizer::pretokenize(std::string_view seg) const {
  std::vector<uint32_t> c = utf8_decode(seg);
  std::vector<std::string> out;
  size_t i = 0;
  while (i < c.size()) {
    size_t len = match_at(c, i);
    if (len == 0) len = 1;      // cannot happen with this pattern; never drop input
    std::string piece;
    for (size_t k = i; k < i + len; ++k) utf8_append(piece, c[k]);
    out.push_back(std::move(piece));
    i += len;
  }
  return out;
}

// ---------------------------------------------------------------- BPE
void Tokenizer::bpe_word(const std::string& w, std::vector<int32_t>& out) const {
  // Initial symbols: one per byte-level codepoint. Each is guaranteed to be in
  // the vocab because the byte alphabet is.
  std::vector<int32_t> sym;
  sym.reserve(w.size());
  for (size_t i = 0; i < w.size();) {
    const size_t start = i;
    utf8_next(w, i);
    auto it = tok_to_id_.find(w.substr(start, i - start));
    if (it == tok_to_id_.end()) return;            // unreachable for byte-level
    sym.push_back(it->second);
  }
  if (sym.empty()) return;

  while (sym.size() > 1) {
    int32_t best_rank = INT32_MAX, best_at = -1, best_id = -1;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      const uint64_t key = (static_cast<uint64_t>(sym[i]) << 32) |
                            static_cast<uint32_t>(sym[i + 1]);
      auto it = merges_.find(key);
      if (it != merges_.end() && it->second.first < best_rank) {
        best_rank = it->second.first;
        best_at   = static_cast<int>(i);
        best_id   = it->second.second;
      }
    }
    if (best_at < 0) break;
    sym[best_at] = best_id;
    sym.erase(sym.begin() + best_at + 1);
  }
  out.insert(out.end(), sym.begin(), sym.end());
}

// ---------------------------------------------------------------- encode
std::vector<int32_t> Tokenizer::encode(std::string_view text, bool allow_special) const {
  std::vector<int32_t> ids;

  // Split on added tokens first: all of them have normalized=false, so they are
  // matched against the RAW text, before NFC.
  size_t pos = 0;
  auto flush_segment = [&](std::string_view seg) {
    if (seg.empty()) return;
    const std::string norm = nfc(seg);
    for (const std::string& piece : pretokenize(norm)) {
      std::string bl;
      bl.reserve(piece.size() * 2);
      for (unsigned char ch : piece) bl += byte_to_uni_[ch];
      bpe_word(bl, ids);
    }
  };

  while (pos < text.size()) {
    size_t hit_at = std::string_view::npos;
    const Added* hit = nullptr;
    if (allow_special) {
      for (const Added& a : added_) {
        const size_t at = text.find(a.content, pos);
        if (at != std::string_view::npos && (hit_at == std::string_view::npos || at < hit_at ||
                                             (at == hit_at && hit &&
                                              a.content.size() > hit->content.size()))) {
          hit_at = at; hit = &a;
        }
      }
    }
    if (!hit) { flush_segment(text.substr(pos)); break; }
    flush_segment(text.substr(pos, hit_at - pos));
    ids.push_back(hit->id);
    pos = hit_at + hit->content.size();
  }
  return ids;
}

// ---------------------------------------------------------------- decode
std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
  std::string bytes;
  bytes.reserve(ids.size() * 4);
  for (int32_t id : ids) {
    auto sp = special_ids_.find(id);
    if (sp != special_ids_.end()) {
      if (skip_special && sp->second) continue;
      // Added tokens are emitted literally; they are not byte-level encoded.
      bytes += id_to_token(id);
      continue;
    }
    const std::string& t = id_to_token(id);
    // Undo the byte-level map.
    for (size_t i = 0; i < t.size();) {
      const uint32_t cp = utf8_next(t, i);
      auto it = uni_to_byte_.find(cp);
      if (it != uni_to_byte_.end()) bytes.push_back(static_cast<char>(it->second));
      else                          utf8_append(bytes, cp);
    }
  }
  return bytes;
}

}  // namespace qwen
