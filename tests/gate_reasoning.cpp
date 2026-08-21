// GATE: the streaming splitter must produce the same content/reasoning/tool
// calls regardless of how the token stream is chopped up.
//
// The interesting failures are all boundary conditions: a tag split across two
// deltas, a delta ending exactly at a tag boundary, back-to-back tool calls.
// So every case is replayed at many different chunk sizes, including 1 byte.
#include <cstdio>
#include <string>
#include <vector>
#include "../src/server/reasoning_parser.h"
#include "../third_party/json.hpp"

struct Case {
  const char* name;
  std::string input;
  std::string want_content;
  std::string want_reasoning;
  std::vector<std::pair<std::string, std::string>> want_calls;  // name, args json
};

int main() {
  std::vector<Case> cases = {
    {"plain", "Hello world", "Hello world", "", {}},
    {"think_only", "<think>reasoning here</think>", "", "reasoning here", {}},
    {"think_then_content", "<think>abc</think>answer", "answer", "abc", {}},
    {"content_then_think", "pre<think>mid</think>post", "prepost", "mid", {}},
    {"unclosed_think", "<think>still going", "", "still going", {}},
    {"tool_simple",
     "<tool_call>\n<function=get_weather>\n<parameter=location>\nMiami\n</parameter>\n"
     "</function>\n</tool_call>",
     "", "", {{"get_weather", R"({"location":"Miami"})"}}},
    {"tool_two_params",
     "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n"
     "<parameter=b>\nhello world\n</parameter>\n</function>\n</tool_call>",
     "", "", {{"f", R"({"a":"1","b":"hello world"})"}}},
    {"tool_json_param",
     "<tool_call>\n<function=g>\n<parameter=obj>\n{\"x\": 1}\n</parameter>\n"
     "</function>\n</tool_call>",
     "", "", {{"g", R"({"obj":{"x":1}})"}}},
    {"think_then_tool",
     "<think>I should call it</think><tool_call>\n<function=h>\n<parameter=k>\nv\n"
     "</parameter>\n</function>\n</tool_call>",
     "", "I should call it", {{"h", R"({"k":"v"})"}}},
    {"text_before_tool",
     "Let me check. <tool_call>\n<function=h>\n<parameter=k>\nv\n</parameter>\n"
     "</function>\n</tool_call>",
     "Let me check. ", "", {{"h", R"({"k":"v"})"}}},
    {"multiline_param",
     "<tool_call>\n<function=w>\n<parameter=code>\nline1\nline2\n</parameter>\n"
     "</function>\n</tool_call>",
     "", "", {{"w", "{\"code\":\"line1\\nline2\"}"}}},
  };

  int fails = 0;
  for (const auto& c : cases) {
    for (int chunk : {1, 2, 3, 5, 7, 13, 64, 100000}) {
      qwen::ReasoningSplitter sp;
      std::string content, reasoning;
      for (size_t i = 0; i < c.input.size(); i += chunk)
        sp.feed(c.input.substr(i, chunk), false, content, reasoning);
      sp.feed("", true, content, reasoning);

      bool ok = (content == c.want_content) && (reasoning == c.want_reasoning) &&
                (sp.tool_calls().size() == c.want_calls.size());
      if (ok) {
        for (size_t k = 0; k < c.want_calls.size(); ++k) {
          if (sp.tool_calls()[k].name != c.want_calls[k].first) { ok = false; break; }
          // compare parsed, so key order and spacing do not matter
          if (nlohmann::json::parse(sp.tool_calls()[k].arguments_json) !=
              nlohmann::json::parse(c.want_calls[k].second)) { ok = false; break; }
        }
      }
      if (!ok) {
        ++fails;
        printf("FAIL %-20s chunk=%-6d\n  content   %s\n  want      %s\n"
               "  reasoning %s\n  want      %s\n  calls %zu (want %zu)\n",
               c.name, chunk, content.c_str(), c.want_content.c_str(),
               reasoning.c_str(), c.want_reasoning.c_str(),
               sp.tool_calls().size(), c.want_calls.size());
        for (const auto& t : sp.tool_calls())
          printf("    got call %s %s\n", t.name.c_str(), t.arguments_json.c_str());
        break;
      }
    }
  }
  printf("gate_reasoning: %d failures over %zu cases x 8 chunk sizes -> %s\n",
         fails, cases.size(), fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
