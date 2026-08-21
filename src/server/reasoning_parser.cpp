#include "reasoning_parser.h"
#include "../../third_party/json.hpp"
#include <algorithm>

namespace qwen {
namespace {
const char* kThinkOpen  = "<think>";
const char* kThinkClose = "</think>";
const char* kToolOpen   = "<tool_call>";
const char* kToolClose  = "</tool_call>";

// Longest tag minus one: how much of the tail must be held back so a tag split
// across two deltas is never emitted as literal text.
size_t hold_bytes() {
  size_t m = 0;
  for (const char* t : {kThinkOpen, kThinkClose, kToolOpen, kToolClose})
    m = std::max(m, std::string(t).size());
  return m - 1;
}
}  // namespace

bool parse_tool_call(const std::string& body, ToolCall& out) {
  const std::string fo = "<function=";
  size_t a = body.find(fo);
  if (a == std::string::npos) return false;
  a += fo.size();
  const size_t b = body.find('>', a);
  if (b == std::string::npos) return false;
  out.name = body.substr(a, b - a);

  nlohmann::ordered_json args = nlohmann::ordered_json::object();
  size_t p = b;
  const std::string po = "<parameter=";
  while ((p = body.find(po, p)) != std::string::npos) {
    p += po.size();
    const size_t q = body.find('>', p);
    if (q == std::string::npos) break;
    const std::string key = body.substr(p, q - p);
    size_t vs = q + 1;
    if (vs < body.size() && body[vs] == '\n') ++vs;
    const size_t ve = body.find("\n</parameter>", vs);
    if (ve == std::string::npos) break;
    std::string val = body.substr(vs, ve - vs);
    // Values arrive as text. Keep JSON-looking values structured so a client
    // that expects a number or an object gets one, but never mangle a string
    // that merely starts with a digit.
    bool structured = false;
    if (!val.empty() && (val[0] == '{' || val[0] == '[' ||
                         val == "true" || val == "false" || val == "null")) {
      try { args[key] = nlohmann::ordered_json::parse(val); structured = true; }
      catch (...) { structured = false; }
    }
    if (!structured) args[key] = val;
    p = ve + 13;
  }
  out.arguments_json = args.dump();
  return true;
}

void ReasoningSplitter::scan_tools(std::string& content) {
  for (;;) {
    if (!in_tool_) {
      const size_t i = content.find(kToolOpen);
      if (i == std::string::npos) return;
      pending_tool_.clear();
      content.erase(i);          // everything from the tag on leaves `content`
      in_tool_ = true;
      // the remainder of this delta was already consumed into buf_ handling
      return;
    }
    return;
  }
}

void ReasoningSplitter::feed(const std::string& delta, bool final,
                             std::string& out_content, std::string& out_reasoning) {
  buf_ += delta;
  for (;;) {
    if (in_tool_) {
      const size_t i = buf_.find(kToolClose);
      if (i == std::string::npos) break;
      pending_tool_ += buf_.substr(0, i);
      buf_.erase(0, i + std::string(kToolClose).size());
      ToolCall tc;
      if (parse_tool_call(pending_tool_, tc)) calls_.push_back(std::move(tc));
      pending_tool_.clear();
      in_tool_ = false;
      continue;
    }
    if (in_think_) {
      const size_t i = buf_.find(kThinkClose);
      if (i == std::string::npos) break;
      out_reasoning += buf_.substr(0, i);
      buf_.erase(0, i + std::string(kThinkClose).size());
      in_think_ = false;
      continue;
    }
    const size_t t = buf_.find(kThinkOpen);
    const size_t c = buf_.find(kToolOpen);
    if (t == std::string::npos && c == std::string::npos) break;
    if (t != std::string::npos && (c == std::string::npos || t < c)) {
      out_content += buf_.substr(0, t);
      buf_.erase(0, t + std::string(kThinkOpen).size());
      in_think_ = true;
    } else {
      out_content += buf_.substr(0, c);
      buf_.erase(0, c + std::string(kToolOpen).size());
      in_tool_ = true;
    }
  }

  const size_t keep = final ? 0 : std::min(hold_bytes(), buf_.size());
  const std::string emit = buf_.substr(0, buf_.size() - keep);
  buf_.erase(0, buf_.size() - keep);
  if (in_tool_)      pending_tool_ += emit;
  else if (in_think_) out_reasoning += emit;
  else                out_content += emit;
}

}  // namespace qwen
