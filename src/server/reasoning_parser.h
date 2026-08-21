// Splits a streaming token feed into `content` and `reasoning_content`, and
// extracts Qwen-format tool calls.
//
// Ported in shape from bonsai-edge-serve's ReasoningSplitter, with three fixes
// its own code needed:
//   * HOLD was strlen("</think>") rather than -1, so a delta ending exactly at
//     a tag boundary emitted a partial tag;
//   * the tags were hardcoded;
//   * there was no preserve_thinking, so reasoning could never be re-injected
//     into the next turn's prompt.
//
// Tool calls use Qwen's XML form, NOT JSON:
//   <tool_call>
//   <function=NAME>
//   <parameter=KEY>
//   VALUE
//   </parameter>
//   </function>
//   </tool_call>
#pragma once
#include <string>
#include <vector>

namespace qwen {

struct ToolCall {
  std::string name;
  std::string arguments_json;   // reassembled as a JSON object string
};

class ReasoningSplitter {
 public:
  explicit ReasoningSplitter(bool starts_inside_think = false)
      : in_think_(starts_inside_think) {}

  // Consumes a decoded delta. Appends to out_content / out_reasoning. `final`
  // flushes whatever is being held back.
  void feed(const std::string& delta, bool final,
            std::string& out_content, std::string& out_reasoning);

  // Tool calls found so far, parsed out of the content stream.
  const std::vector<ToolCall>& tool_calls() const { return calls_; }
  bool in_think() const { return in_think_; }

 private:
  void scan_tools(std::string& content);
  std::string buf_;
  std::string pending_tool_;
  bool in_think_ = false;
  bool in_tool_ = false;
  std::vector<ToolCall> calls_;
};

// Parses one <tool_call>...</tool_call> body into a name and a JSON object.
bool parse_tool_call(const std::string& body, ToolCall& out);

}  // namespace qwen
