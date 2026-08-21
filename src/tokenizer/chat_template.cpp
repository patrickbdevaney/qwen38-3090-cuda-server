#include "chat_template.h"
#include "unicode.h"

#include <string_view>
#include <vector>

namespace qwen {
namespace {

// Jinja's |trim is Python str.strip(), which uses str.isspace(). That set is
// Unicode White_Space PLUS U+001C..U+001F, which are not White_Space. Matching
// it exactly matters: a trim that differs by one character shifts every
// subsequent byte of the render.
inline bool py_isspace(uint32_t cp) {
  return is_space(cp) || (cp >= 0x1C && cp <= 0x1F);
}

std::string trim(std::string_view s) {
  std::vector<uint32_t> c = utf8_decode(s);
  size_t a = 0, b = c.size();
  while (a < b && py_isspace(c[a])) ++a;
  while (b > a && py_isspace(c[b - 1])) --b;
  std::string out;
  for (size_t i = a; i < b; ++i) utf8_append(out, c[i]);
  return out;
}

bool starts_with(const std::string& s, const char* p) {
  return s.rfind(p, 0) == 0;
}
bool ends_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

// JSON string escaping as Python's json.dumps does it with ensure_ascii=False:
// escape only ", \\ and the C0 controls; emit everything else as raw UTF-8.
void dump_string(const std::string& s, std::string& out) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"':  out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n";  break;
      case '\r': out += "\\r";  break;
      case '\t': out += "\\t";  break;
      case '\b': out += "\\b";  break;
      case '\f': out += "\\f";  break;
      default:
        if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); out += b; }
        else            out += static_cast<char>(c);
    }
  }
  out += '"';
}

void dump(const ojson& j, std::string& out) {
  if (j.is_object()) {
    out += '{';
    bool first = true;
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (!first) out += ", ";              // Python's default item separator
      first = false;
      dump_string(it.key(), out);
      out += ": ";                          // Python's default key separator
      dump(it.value(), out);
    }
    out += '}';
  } else if (j.is_array()) {
    out += '[';
    bool first = true;
    for (const auto& v : j) {
      if (!first) out += ", ";
      first = false;
      dump(v, out);
    }
    out += ']';
  } else if (j.is_string()) {
    dump_string(j.get<std::string>(), out);
  } else if (j.is_boolean()) {
    out += j.get<bool>() ? "true" : "false";
  } else if (j.is_null()) {
    out += "null";
  } else {
    out += j.dump();                        // numbers
  }
}

// render_content(): a message's content is either a string or an array of
// parts. v1 is text-only, so image/video parts raise rather than emitting the
// vision placeholders the template would.
// Set for the duration of a render_chat() call. The template's own vision
// branch emits these three tokens verbatim; whether we are allowed to follow it
// is a build/launch decision, not a template one.
bool g_allow_vision = false;

std::string render_content(const ojson& content) {
  if (content.is_null()) return "";
  if (content.is_string()) return content.get<std::string>();
  if (content.is_array()) {
    std::string out;
    for (const auto& item : content) {
      if (!item.is_object()) throw ChatTemplateError("Unexpected item type in content.");
      const std::string type = item.value("type", "");
      if (item.contains("image") || item.contains("image_url") || type == "image") {
        if (!g_allow_vision)
          throw UnsupportedContent("image content is not supported: start the server with --vision");
        out += "<|vision_start|><|image_pad|><|vision_end|>";
        continue;
      }
      if (item.contains("video") || type == "video")
        throw UnsupportedContent("video content is not supported");
      if (item.contains("text")) out += item.at("text").get<std::string>();
      else throw ChatTemplateError("Unexpected item type in content.");
    }
    return out;
  }
  throw ChatTemplateError("Unexpected content type.");
}

const char* kToolFormat =
  "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
  "\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
  "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
  "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>"
  "\n\n<IMPORTANT>\nReminder:\n"
  "- Function calls MUST follow the specified format: an inner <function=...></function> block "
  "must be nested within <tool_call></tool_call> XML tags\n"
  "- Required parameters MUST be specified\n"
  "- You may provide optional reasoning for your function call in natural language BEFORE the "
  "function call, but NOT after\n"
  "- If there is no function call available, answer the question like normal with your current "
  "knowledge and do not tell the user about function calls\n</IMPORTANT>";

}  // namespace

std::string json_dumps_python(const ojson& j) {
  std::string out;
  dump(j, out);
  return out;
}

std::string render_chat(const ojson& messages, const ojson& tools,
                        const ChatOptions& opt) {
  g_allow_vision = opt.allow_vision;
  if (!messages.is_array() || messages.empty())
    throw ChatTemplateError("No messages provided.");

  // ---- reasoning instructions ----------------------------------------
  std::string reasoning_instructions;
  const bool thinking_on = !opt.enable_thinking.has_value() || *opt.enable_thinking;
  if (thinking_on) {
    const std::string eff = opt.reasoning_effort.empty() ? "xhigh" : opt.reasoning_effort;
    if (eff != "xhigh" && eff != "medium" && eff != "low")
      throw ChatTemplateError("Unexpected reasoning effort " + eff +
                              ". Supported types are xhigh (default), medium, and low.");
    if (eff == "xhigh")
      reasoning_instructions =
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate "
        "key assumptions, consider plausible alternatives, and prioritize correctness, "
        "consistency, and clarity in the final answer.";
    else if (eff == "low")
      reasoning_instructions =
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly "
        "to the conclusion without unnecessary elaboration.";
    // "medium" deliberately injects nothing.
  }

  std::string out;
  const bool has_tools = tools.is_array() && !tools.empty();
  const bool sys_first = messages[0].contains("role") &&
                         messages[0]["role"] == "system";

  // ---- system / tools preamble ---------------------------------------
  if (has_tools) {
    out += "<|im_start|>system\n";
    if (!reasoning_instructions.empty()) out += reasoning_instructions + "\n\n";
    out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const auto& t : tools) { out += "\n"; out += json_dumps_python(t); }
    out += "\n</tools>";
    out += kToolFormat;
    if (sys_first) {
      const std::string c = trim(render_content(messages[0].value("content", ojson())));
      if (!c.empty()) out += "\n\n" + c;
    }
    out += "<|im_end|>\n";
  } else if (sys_first) {
    const std::string c = trim(render_content(messages[0].value("content", ojson())));
    if (!c.empty())
      out += "<|im_start|>system\n" +
             (reasoning_instructions.empty() ? "" : reasoning_instructions + "\n\n") +
             c + "<|im_end|>\n";
    else if (!reasoning_instructions.empty())
      out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
  } else if (!reasoning_instructions.empty()) {
    out += "<|im_start|>system\n" + reasoning_instructions + "<|im_end|>\n";
  }

  // ---- last_query_index ----------------------------------------------
  // Scan backward for the last user turn that is a real query rather than a
  // wrapped tool result. This is what preserve_thinking=false keys off.
  bool multi_step_tool = true;
  size_t last_query_index = messages.size() - 1;
  for (size_t k = messages.size(); k-- > 0;) {
    const auto& m = messages[k];
    if (multi_step_tool && m.contains("role") && m["role"] == "user") {
      const std::string c = trim(render_content(m.value("content", ojson())));
      if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) {
        multi_step_tool = false;
        last_query_index = k;
      }
    }
  }
  if (multi_step_tool) throw ChatTemplateError("No user query found in messages.");

  // ---- main loop ------------------------------------------------------
  for (size_t i = 0; i < messages.size(); ++i) {
    const auto& m = messages[i];
    const std::string role = m.value("role", "");
    const std::string content = trim(render_content(m.value("content", ojson())));

    if (role == "system") {
      if (i != 0) throw ChatTemplateError("System message must be at the beginning.");
      // already emitted above
    } else if (role == "user") {
      out += "<|im_start|>user\n" + content + "<|im_end|>\n";
    } else if (role == "assistant") {
      std::string reasoning;
      if (m.contains("reasoning_content") && m["reasoning_content"].is_string())
        reasoning = trim(m["reasoning_content"].get<std::string>());

      const bool keep_think = !opt.preserve_thinking.has_value() ||
                              *opt.preserve_thinking || i > last_query_index;
      if (keep_think)
        out += "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content;
      else
        out += "<|im_start|>assistant\n" + content;

      if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
        bool first = true;
        for (const auto& tc_raw : m["tool_calls"]) {
          const ojson& tc = tc_raw.contains("function") ? tc_raw["function"] : tc_raw;
          const std::string name = tc.value("name", "");
          if (first) {
            out += (trim(content).empty() ? "" : "\n\n");
            out += "<tool_call>\n<function=" + name + ">\n";
          } else {
            out += "\n<tool_call>\n<function=" + name + ">\n";
          }
          first = false;
          if (tc.contains("arguments") && !(tc["arguments"].is_string() &&
                                            tc["arguments"].get<std::string>().empty())) {
            const ojson& args = tc["arguments"];
            if (!args.is_object())
              throw ChatTemplateError(
                  "tool_call.function.arguments must be an object; OpenAI clients send a JSON "
                  "string, so the server must parse it before rendering");
            for (auto it = args.begin(); it != args.end(); ++it) {
              out += "<parameter=" + it.key() + ">\n";
              out += it.value().is_string() ? it.value().get<std::string>()
                                            : json_dumps_python(it.value());
              out += "\n</parameter>\n";
            }
          }
          out += "</function>\n</tool_call>";
        }
      }
      out += "<|im_end|>\n";
    } else if (role == "tool") {
      const bool prev_is_tool = i > 0 && messages[i - 1].value("role", "") == "tool";
      if (i > 0 && !prev_is_tool) out += "<|im_start|>user";
      out += "\n<tool_response>\n" + content + "\n</tool_response>";
      const bool last = (i + 1 == messages.size());
      const bool next_is_tool = !last && messages[i + 1].value("role", "") == "tool";
      if (last || !next_is_tool) out += "<|im_end|>\n";
    } else {
      throw ChatTemplateError("Unexpected message role.");
    }
  }

  // ---- generation prompt ----------------------------------------------
  if (opt.add_generation_prompt) {
    out += "<|im_start|>assistant\n";
    // Only an EXPLICIT enable_thinking=false pre-closes the block.
    if (opt.enable_thinking.has_value() && !*opt.enable_thinking)
      out += "<think>\n\n</think>\n\n";
    else
      out += "<think>\n";
  }
  return out;
}

}  // namespace qwen
