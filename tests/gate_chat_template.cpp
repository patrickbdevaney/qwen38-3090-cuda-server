// GATE (directive P1): chat template output must match HF apply_chat_template
// BYTE FOR BYTE across 30 fixtures covering thinking on/off, preserve_thinking
// on/off, reasoning_effort, tool declarations, tool-call round trips,
// multi-turn and system prompts.
//
// Also checks that tokenizing our render reproduces the reference ids, so a
// template bug cannot hide behind a tokenizer that happens to agree.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "../src/tokenizer/chat_template.h"
#include "../src/tokenizer/bpe.h"

using qwen::ojson;

static void show_diff(const std::string& want, const std::string& got) {
  size_t k = 0;
  while (k < want.size() && k < got.size() && want[k] == got[k]) ++k;
  auto win = [&](const std::string& s) {
    const size_t a = k > 40 ? k - 40 : 0;
    std::string o = s.substr(a, 90), e;
    for (char c : o) { if (c == '\n') e += "\\n"; else if (c == '\t') e += "\\t"; else e += c; }
    return e;
  };
  printf("    first difference at byte %zu (hf %zu bytes, ours %zu)\n", k, want.size(), got.size());
  printf("    hf  : ...%s...\n", win(want).c_str());
  printf("    ours: ...%s...\n", win(got).c_str());
}

int main(int argc, char** argv) {
  const std::string fixtures = argc > 1 ? argv[1] : "tests/fixtures/chat/fixtures.jsonl";
  const std::string tokjson  = argc > 2 ? argv[2]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ/tokenizer.json";

  qwen::Tokenizer tk;
  bool have_tok = true;
  try { tk.load(tokjson); }
  catch (const std::exception& e) {
    printf("  (tokenizer unavailable, checking renders only: %s)\n", e.what());
    have_tok = false;
  }

  std::ifstream f(fixtures);
  if (!f) { fprintf(stderr, "cannot open %s\n", fixtures.c_str()); return 2; }

  size_t n = 0, render_fail = 0, id_fail = 0;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    ojson j = ojson::parse(line);
    const std::string name = j["name"].get<std::string>();
    const std::string want = j["render"].get<std::string>();
    const ojson& msgs = j["messages"];
    const ojson& kw   = j["kwargs"];

    qwen::ChatOptions opt;
    opt.add_generation_prompt = kw.contains("add_generation_prompt") &&
                                kw["add_generation_prompt"].get<bool>();
    if (kw.contains("enable_thinking"))   opt.enable_thinking   = kw["enable_thinking"].get<bool>();
    if (kw.contains("preserve_thinking")) opt.preserve_thinking = kw["preserve_thinking"].get<bool>();
    if (kw.contains("reasoning_effort"))  opt.reasoning_effort  = kw["reasoning_effort"].get<std::string>();
    const ojson tools = kw.contains("tools") ? kw["tools"] : ojson();

    ++n;
    std::string got;
    try { got = qwen::render_chat(msgs, tools, opt); }
    catch (const std::exception& e) {
      printf("  [%s] THREW: %s\n", name.c_str(), e.what());
      ++render_fail; continue;
    }
    if (got != want) {
      printf("  RENDER MISMATCH [%s]\n", name.c_str());
      show_diff(want, got);
      ++render_fail; continue;
    }
    if (have_tok && j.contains("ids")) {
      const std::vector<int32_t> want_ids = j["ids"].get<std::vector<int32_t>>();
      const std::vector<int32_t> got_ids  = tk.encode(got);
      if (got_ids != want_ids) {
        printf("  IDS MISMATCH [%s]: hf %zu ids, ours %zu\n",
               name.c_str(), want_ids.size(), got_ids.size());
        ++id_fail;
      }
    }
  }

  printf("\ngate_chat_template\n");
  printf("  fixtures        : %zu\n", n);
  printf("  render failures : %zu\n", render_fail);
  printf("  id failures     : %zu\n", id_fail);
  const bool pass = !render_fail && !id_fail && n >= 30;
  printf("  RESULT          : %s%s\n", pass ? "PASS" : "FAIL",
         n < 30 ? "  (directive asks for 30 fixtures)" : "");
  return pass ? 0 : 1;
}
