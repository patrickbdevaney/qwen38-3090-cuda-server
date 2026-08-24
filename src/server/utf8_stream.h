#pragma once
// Streaming-safe UTF-8 boundaries.
//
// WHY THIS EXISTS
//
// Qwen's tokenizer is byte-level BPE, so a single token can carry *half* of a
// multi-byte character. The SSE path serialises one token per chunk, and
// nlohmann::json refuses to serialise a std::string that is not valid UTF-8 --
// it throws type_error.316. That exception escaped the chunked content provider
// and reached std::terminate, which killed the whole server:
//
//   terminate called after throwing an instance of nlohmann::detail::type_error
//     what(): [json.exception.type_error.316] incomplete UTF-8 string; last byte: 0x80
//
// Observed 2026-08-23 killing a live agent session mid-run. Any prompt whose
// output contains a curly quote, an em dash, a box-drawing character or an
// emoji can trigger it -- which is to say, ordinary output.
//
// The fix is not to catch the exception. Emitting half a character to a client
// is wrong regardless of whether the serialiser objects: the client concatenates
// deltas and would render a replacement character in the middle of a word. So
// the stream holds a trailing incomplete sequence back until the next token
// completes it. That is also what makes the fix free -- the byte is only
// delayed by one token, never lost.
#include <string>
#include "../../third_party/json.hpp"

namespace qwen {

// Byte length of the character introduced by lead byte `c`.
// A stray continuation byte or invalid lead consumes exactly one byte, so a
// corrupt stream still makes forward progress instead of stalling.
inline size_t utf8_char_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c & 0xE0) == 0xC0) return 2;
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xF8) == 0xF0) return 4;
  return 1;
}

// Offset of the first byte of the final (possibly incomplete) character.
// Scans back at most 4 bytes: no valid encoding is longer, so a run of
// continuation bytes beyond that is garbage and is passed through as-is.
inline size_t utf8_last_char_start(const std::string& s) {
  size_t i = s.size();
  for (int steps = 0; i > 0 && steps < 4; ++steps) {
    --i;
    if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) return i;
  }
  return s.size();
}

// Append `piece` to `carry` and return the longest prefix that ends on a
// complete character. Whatever remains stays in `carry` for the next token.
inline std::string utf8_flush(std::string& carry, const std::string& piece) {
  carry += piece;
  if (carry.empty()) return std::string();
  const size_t start = utf8_last_char_start(carry);
  if (start >= carry.size()) { std::string out; out.swap(carry); return out; }
  const size_t need = utf8_char_len(static_cast<unsigned char>(carry[start]));
  if (carry.size() - start >= need) { std::string out; out.swap(carry); return out; }
  std::string out = carry.substr(0, start);
  carry.erase(0, start);
  return out;
}

// End of generation with bytes still held: the sequence can never complete, so
// surface it as U+FFFD rather than dropping it silently or emitting raw bytes.
inline std::string utf8_flush_tail(std::string& carry) {
  if (carry.empty()) return std::string();
  carry.clear();
  return std::string("\xEF\xBF\xBD");
}

// Backstop. Even with boundaries handled, a tool-call argument string or a
// model-emitted field can still contain invalid bytes, and no serialisation on
// the response path is worth terminating the process over.
inline std::string json_dump_safe(const nlohmann::json& j) {
  return j.dump(-1, 32, false, nlohmann::json::error_handler_t::replace);
}

}  // namespace qwen
