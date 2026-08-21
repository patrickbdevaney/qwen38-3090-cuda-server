// Qwen3.8 chat template, hand-ported from chat_template.jinja.
//
// Gated byte-for-byte against HF apply_chat_template across 30 fixtures
// (tests/gate_chat_template.cpp). It is a hand port rather than a Jinja
// interpreter because the serving path must not carry one, and because the
// template's behaviour is a fixed contract for this checkpoint.
//
// Semantics that are easy to get wrong and are covered by fixtures:
//   - reasoning_effort defaults to "xhigh"; "medium" injects NOTHING.
//   - preserve_thinking=false strips <think> only from assistant turns at or
//     before ns.last_query_index, which is found by scanning BACKWARD for the
//     last user turn whose content is not a bare <tool_response> wrapper.
//   - tool results render as a *user* turn containing <tool_response>, and
//     consecutive tool messages share one <|im_start|>user block.
//   - tool_calls render as Qwen XML (<function=NAME><parameter=K>), not JSON.
//   - tools are serialised with Python json.dumps separators (", " and ": ")
//     and insertion order, not nlohmann's compact form.
#pragma once
#include <optional>
#include <stdexcept>
#include <string>
#include "../../third_party/json.hpp"

namespace qwen {

using ojson = nlohmann::ordered_json;

struct ChatOptions {
  bool add_generation_prompt = false;
  std::optional<bool> enable_thinking;    // unset == Jinja "undefined"
  std::optional<bool> preserve_thinking;  // unset == Jinja "undefined"
  std::string reasoning_effort;           // "" == undefined -> "xhigh"
  // When true, image parts render as <|vision_start|><|image_pad|><|vision_end|>
  // exactly as the Jinja template does, and the caller is responsible for
  // expanding each <|image_pad|> into one token per image patch-block. When
  // false (the default) an image part raises UnsupportedContent.
  bool allow_vision = false;
};

// Thrown for content the template supports but this build does not (images,
// video). The server turns this into a 400 with a clear message; v1 is
// text-only by design (see README).
struct UnsupportedContent : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Thrown for the template's own raise_exception() cases.
struct ChatTemplateError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// `tools` may be null/absent.
std::string render_chat(const ojson& messages, const ojson& tools,
                        const ChatOptions& opt);

// Exposed for tests: Python json.dumps(obj) with default separators,
// ensure_ascii=False, insertion order preserved.
std::string json_dumps_python(const ojson& j);

}  // namespace qwen
