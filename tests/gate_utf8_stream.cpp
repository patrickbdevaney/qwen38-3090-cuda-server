// GATE: a streamed response must never contain half a UTF-8 character.
//
// This gate exists because the absence of it cost a live server. Byte-level BPE
// splits multi-byte characters across tokens; nlohmann::json throws
// type_error.316 on an incomplete string, and that exception escaping the SSE
// content provider reached std::terminate and killed the process.
//
// Two properties are checked, and the second is the one that matters:
//   1. every emitted chunk is independently valid UTF-8, and
//   2. concatenating every chunk reproduces the input byte for byte.
// A splitter that satisfies (1) by dropping bytes would be worse than the crash.
#include <cstdio>
#include <string>
#include <vector>
#include "../src/server/utf8_stream.h"

using qwen::utf8_flush;
using qwen::utf8_flush_tail;
using nlohmann::json;

static bool valid_utf8(const std::string& s) {
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3
             : (c & 0xF8) == 0xF0 ? 4 : 0;
    if (n == 0 || i + n > s.size()) return false;
    for (size_t k = 1; k < n; ++k)
      if ((static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) return false;
    i += n;
  }
  return true;
}

static int fails = 0;
static void ck(const char* name, bool ok) {
  printf("  %s %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) ++fails;
}

// Feed `text` one byte at a time -- the worst case, and the one that used to
// crash: every multi-byte character is split at every possible offset.
static void round_trip(const char* name, const std::string& text) {
  std::string carry, joined;
  bool all_valid = true;
  for (char ch : text) {
    std::string out = utf8_flush(carry, std::string(1, ch));
    if (!out.empty()) {
      if (!valid_utf8(out)) all_valid = false;
      // The real failure mode: this is what the server hands to the serialiser.
      json d;
      d["content"] = out;
      try { (void)d.dump(); } catch (...) { all_valid = false; }
      joined += out;
    }
  }
  joined += utf8_flush_tail(carry);
  ck(name, all_valid && joined == text);
}

int main() {
  round_trip("ascii round-trips",            "hello world");
  round_trip("curly quote (2-byte)",         "it\xE2\x80\x99s fine");
  round_trip("em dash + box drawing",        "a \xE2\x80\x94 b \xE2\x94\x82 c");
  round_trip("cjk (3-byte)",                 "\xE4\xBD\xA0\xE5\xA5\xBD, world");
  round_trip("emoji (4-byte)",               "ship it \xF0\x9F\x9A\x80 now");
  round_trip("adjacent multibyte",           "\xC3\xA9\xC3\xA8\xC3\xAA\xC3\xAB");

  // A lone lead byte can never complete; it must surface as U+FFFD, and it must
  // not be silently dropped or emitted raw.
  {
    std::string carry;
    std::string out = utf8_flush(carry, "abc\xE2\x80");
    std::string tail = utf8_flush_tail(carry);
    ck("truncated tail held back", out == "abc");
    ck("truncated tail becomes U+FFFD", tail == "\xEF\xBF\xBD" && valid_utf8(tail));
  }

  // Whole characters must pass straight through without latency.
  {
    std::string carry;
    ck("complete input is not delayed", utf8_flush(carry, "done\xE2\x80\x99") == "done\xE2\x80\x99"
       && carry.empty());
  }

  // A stray continuation byte must make forward progress, not stall the stream.
  {
    std::string carry;
    std::string out = utf8_flush(carry, "\x80\x80ok");
    ck("garbage still progresses", out.size() == 4);
  }

  printf(fails ? "gate_utf8_stream FAILED (%d)\n" : "gate_utf8_stream ok\n", fails);
  return fails ? 1 : 0;
}
