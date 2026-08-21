// UTF-8 handling, Unicode general categories, and NFC normalization.
//
// Everything here exists because the Qwen tokenizer needs it and the C++
// standard library does not provide it:
//   - the pretokenizer regex uses \p{L}, \p{M}, \p{N} and a Unicode-aware \s,
//     none of which std::regex can express;
//   - the tokenizer's normalizer is NFC, and it is LOSSY: HF's
//     decode(encode("cafe" + U+0301)) returns the precomposed form. Skipping
//     NFC therefore changes token ids on any text with combining marks.
//
// Tables come from tools/gen_unicode_tables.py (Unicode 15.1.0).
#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include "unicode_tables.h"

namespace qwen {

// ---------------------------------------------------------------- UTF-8
// Decoding is deliberately permissive in one direction only: invalid bytes
// become U+FFFD so that a malformed prompt cannot crash the server. It never
// silently drops input, because a dropped byte would shift every downstream
// token id.
inline uint32_t utf8_next(std::string_view s, size_t& i) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  auto cont = [&](size_t k) {
    return i + k < s.size() && (static_cast<unsigned char>(s[i + k]) & 0xC0) == 0x80;
  };
  if (c < 0x80)                       { i += 1; return c; }
  if ((c & 0xE0) == 0xC0 && cont(1))  { uint32_t cp = ((c & 0x1Fu) << 6)
                                        | (static_cast<unsigned char>(s[i+1]) & 0x3Fu);
                                        i += 2; return cp < 0x80 ? 0xFFFD : cp; }
  if ((c & 0xF0) == 0xE0 && cont(1) && cont(2)) {
      uint32_t cp = ((c & 0x0Fu) << 12)
                  | ((static_cast<unsigned char>(s[i+1]) & 0x3Fu) << 6)
                  |  (static_cast<unsigned char>(s[i+2]) & 0x3Fu);
      i += 3; return (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) ? 0xFFFD : cp; }
  if ((c & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
      uint32_t cp = ((c & 0x07u) << 18)
                  | ((static_cast<unsigned char>(s[i+1]) & 0x3Fu) << 12)
                  | ((static_cast<unsigned char>(s[i+2]) & 0x3Fu) << 6)
                  |  (static_cast<unsigned char>(s[i+3]) & 0x3Fu);
      i += 4; return (cp < 0x10000 || cp > 0x10FFFF) ? 0xFFFD : cp; }
  i += 1; return 0xFFFD;
}

inline void utf8_append(std::string& out, uint32_t cp) {
  if (cp < 0x80) { out.push_back(static_cast<char>(cp)); return; }
  if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); return; }
  if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F))); return; }
  out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
  out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
  out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
  out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
}

inline std::vector<uint32_t> utf8_decode(std::string_view s) {
  std::vector<uint32_t> cps;
  cps.reserve(s.size());
  for (size_t i = 0; i < s.size();) cps.push_back(utf8_next(s, i));
  return cps;
}

inline std::string utf8_encode(const std::vector<uint32_t>& cps) {
  std::string out;
  out.reserve(cps.size() * 2);
  for (uint32_t cp : cps) utf8_append(out, cp);
  return out;
}

// ---------------------------------------------------------------- categories
inline bool in_ranges(const uint32_t (*tbl)[2], size_t n, uint32_t cp) {
  size_t lo = 0, hi = n;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < tbl[mid][0])      hi = mid;
    else if (cp > tbl[mid][1]) lo = mid + 1;
    else                       return true;
  }
  return false;
}

inline bool is_letter(uint32_t cp) { return in_ranges(uni::kLetter, uni::kLetter_N, cp); }
inline bool is_mark  (uint32_t cp) { return in_ranges(uni::kMark,   uni::kMark_N,   cp); }
inline bool is_number(uint32_t cp) { return in_ranges(uni::kNumber, uni::kNumber_N, cp); }
inline bool is_space (uint32_t cp) { return in_ranges(uni::kWhitespace, uni::kWhitespace_N, cp); }

// ---------------------------------------------------------------- NFC
inline uint8_t ccc(uint32_t cp) {
  size_t lo = 0, hi = uni::kCcc_N;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < uni::kCcc[mid].cp)      hi = mid;
    else if (cp > uni::kCcc[mid].cp) lo = mid + 1;
    else                             return uni::kCcc[mid].ccc;
  }
  return 0;
}

// Hangul syllables decompose and compose algorithmically; keeping them out of
// the tables makes kDecomp about an order of magnitude smaller.
constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28;
constexpr uint32_t kNCount = kVCount * kTCount;          // 588
constexpr uint32_t kSCount = kLCount * kNCount;          // 11172

inline void decompose_cp(uint32_t cp, std::vector<uint32_t>& out) {
  if (cp >= kSBase && cp < kSBase + kSCount) {           // Hangul
    uint32_t s = cp - kSBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    if (uint32_t t = s % kTCount) out.push_back(kTBase + t);
    return;
  }
  size_t lo = 0, hi = uni::kDecomp_N;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < uni::kDecomp[mid].cp)      hi = mid;
    else if (cp > uni::kDecomp[mid].cp) lo = mid + 1;
    else {
      decompose_cp(uni::kDecomp[mid].a, out);            // recursive: canonical
      if (uni::kDecomp[mid].b) decompose_cp(uni::kDecomp[mid].b, out);
      return;
    }
  }
  out.push_back(cp);
}

inline uint32_t compose_pair(uint32_t a, uint32_t b) {
  // Hangul, algorithmic
  if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount)
    return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
  if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 &&
      b > kTBase && b < kTBase + kTCount)
    return a + (b - kTBase);
  size_t lo = 0, hi = uni::kComp_N;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    const auto& e = uni::kComp[mid];
    if (e.a < a || (e.a == a && e.b < b))      lo = mid + 1;
    else if (e.a > a || (e.a == a && e.b > b)) hi = mid;
    else                                       return e.c;
  }
  return 0;
}

// Standard NFC: canonical decomposition, canonical ordering, canonical
// composition. Quick-check first, because the overwhelming majority of real
// prompts are already NFC and copying them is wasted work on the hot path.
inline std::string nfc(std::string_view s) {
  bool ascii_only = true;
  for (unsigned char c : s) if (c >= 0x80) { ascii_only = false; break; }
  if (ascii_only) return std::string(s);

  std::vector<uint32_t> cps = utf8_decode(s);

  // D: canonical decomposition
  std::vector<uint32_t> d;
  d.reserve(cps.size() + 8);
  for (uint32_t cp : cps) decompose_cp(cp, d);

  // O: canonical ordering (stable insertion sort within each combining run)
  for (size_t i = 1; i < d.size(); ++i) {
    uint8_t c = ccc(d[i]);
    if (c == 0) continue;
    size_t j = i;
    while (j > 0) {
      uint8_t p = ccc(d[j - 1]);
      if (p == 0 || p <= c) break;
      std::swap(d[j], d[j - 1]);
      --j;
    }
  }

  // C: canonical composition
  std::vector<uint32_t> out;
  out.reserve(d.size());
  size_t starter = SIZE_MAX;      // index in `out` of the current starter
  int last_ccc = -1;              // ccc of the last char appended after it
  for (uint32_t cp : d) {
    uint8_t c = ccc(cp);
    if (starter != SIZE_MAX && (last_ccc < static_cast<int>(c) || last_ccc == -1)) {
      if (uint32_t comp = compose_pair(out[starter], cp)) {
        out[starter] = comp;
        continue;                 // last_ccc unchanged: blocked set is unaffected
      }
    }
    if (c == 0) { starter = out.size(); last_ccc = -1; }
    else        { last_ccc = static_cast<int>(c); }
    out.push_back(cp);
  }
  return utf8_encode(out);
}

}  // namespace qwen
