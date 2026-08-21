#!/usr/bin/env python3
"""
PHASE 1 ORACLE — tokenizer and chat-template ground truth.

Emits fixtures that the C++ implementation in src/tokenizer/ must reproduce
exactly.  This runs once, from HF transformers, and the output is committed;
the C++ gate never calls Python.

Two fixture sets:
  tests/fixtures/tokenizer/corpus.jsonl  segments of mixed code/prose/CJK plus
                                         adversarial cases, with expected ids
                                         and expected decoded text
  tests/fixtures/chat/fixtures.jsonl     30 chat-template renders covering
                                         thinking on/off, preserve_thinking
                                         on/off, reasoning_effort, tools,
                                         multi-turn and system prompts

Gate (directive P1): >=100k tokens round-trip exactly, and every chat render
matches HF apply_chat_template byte for byte.

Usage: python3 tools/dump_tokenizer_ref.py <model_dir> [--out tests/fixtures]
"""
import argparse, json, os, random, sys, unicodedata

def read(p, limit=None):
    try:
        with open(p, "r", encoding="utf-8", errors="strict") as f:
            s = f.read()
        return s[:limit] if limit else s
    except (OSError, UnicodeDecodeError):
        return ""

# ---------------------------------------------------------------- corpus
def build_corpus(tk, repo_root, prior_art):
    """Returns [(name, text)].  Aims well past 100k tokens with broad coverage."""
    segs = []

    # -- 1. real source code: C++/CUDA/Python/CMake/shell/JSON ------------
    code_sources = []
    for root in (repo_root, prior_art):
        if not os.path.isdir(root): continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in
                           (".git", "build", "node_modules", "__pycache__", "models")]
            for fn in filenames:
                if fn.endswith((".cu", ".cuh", ".cpp", ".h", ".hpp", ".c",
                                ".py", ".sh", ".txt", ".json", ".yaml", ".cc")):
                    # Skip machine-generated, near-duplicate blobs: they add mass
                    # without adding coverage, and would let a weak tokenizer pass
                    # the token-count gate on repetitive input.
                    if fn.startswith("model-") and fn.endswith(".safetensors.json"): continue
                    if "index.json" in fn or fn.endswith(".lock"): continue
                    p = os.path.join(dirpath, fn)
                    try:
                        sz = os.path.getsize(p)
                        if sz > 400_000 or sz < 300: continue
                    except OSError: continue
                    code_sources.append(p)
    random.Random(0).shuffle(code_sources)
    for i, p in enumerate(code_sources[:130]):
        t = read(p, 9_000)
        if len(t) > 200:
            segs.append((f"code_{i:02d}_{os.path.basename(p)}", t))

    # -- 2. real prose / markdown ----------------------------------------
    md = []
    for root in (repo_root, prior_art):
        if not os.path.isdir(root): continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", "build")]
            md += [os.path.join(dirpath, f) for f in filenames if f.endswith(".md")]
    random.Random(1).shuffle(md)
    for i, p in enumerate(md[:70]):
        t = read(p, 9_000)
        if len(t) > 200:
            segs.append((f"prose_{i:02d}_{os.path.basename(p)}", t))

    # -- 3. multi-script coverage, synthesized from the vocabulary --------
    # Decoding real vocabulary entries is *more* adversarial than natural text:
    # it exercises rare tokens, CJK, Cyrillic, Arabic, Devanagari and emoji that
    # a natural sample would never reach.
    vocab_size = tk.vocab_size
    rng = random.Random(2)
    def script_of(ch):
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF: return "han"
        if 0x3040 <= o <= 0x30FF: return "kana"
        if 0xAC00 <= o <= 0xD7AF: return "hangul"
        if 0x0400 <= o <= 0x04FF: return "cyrillic"
        if 0x0600 <= o <= 0x06FF: return "arabic"
        if 0x0900 <= o <= 0x097F: return "devanagari"
        if 0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF: return "emoji"
        return None
    buckets = {}
    for tid in range(vocab_size):
        s = tk.decode([tid])
        if not s: continue
        sc = next((script_of(c) for c in s if script_of(c)), None)
        if sc: buckets.setdefault(sc, []).append(s)
    for sc, items in sorted(buckets.items()):
        rng.shuffle(items)
        chunk = items[:1200]
        # join in runs, with punctuation and spacing that exercises the
        # pretokenizer's script-boundary handling
        text = ""
        for j, piece in enumerate(chunk):
            text += piece
            if j % 7 == 6:  text += "。\n" if sc in ("han","kana") else ".\n"
            elif j % 3 == 2: text += " "
        if text.strip():
            segs.append((f"script_{sc}", text))

    # -- 4. adversarial ---------------------------------------------------
    adv = {
        "adv_all_bytes": "".join(chr(b) for b in range(1, 256)),
        "adv_ascii_control": "".join(chr(b) for b in range(1, 32)) + "\x7f",
        "adv_whitespace_runs": "a" + " "*1 + "b" + " "*2 + "c" + " "*7 + "d\n\n\n\te\r\nf\r\n\r\ng" + " "*20,
        "adv_trailing_space": "word ",
        "adv_leading_space": " word",
        "adv_only_spaces": "     ",
        "adv_newline_only": "\n\n\n",
        "adv_contractions": "It's Bob's. They're. I've. I'm. We'll. He'd. IT'S BOB'S. THEY'RE.",
        "adv_numbers": "0 1 12 123 1234 12345 007 3.14159 1e-9 0x1F 1_000_000 -42 +42 ٠١٢٣٤ 一二三",
        "adv_combining_nfd": "café école naïve Å ẛ̣",
        "adv_combining_nfc": unicodedata.normalize("NFC", "café école naïve Å ẛ̣"),
        "adv_nfkc_sensitive": "ﬁ ﬂ ½ ² ㍿ ﷺ Ⅻ ｱｲｳ",
        "adv_zero_width": "a​b‌c‍d﻾⁠e",
        "adv_rtl": "مرحبا بالعالم! שלום עולם. mixed العربية text",
        "adv_emoji_zwj": "👨‍👩‍👧‍👦 👍🏽 🇺🇸 🏳️‍🌈 ❤️ 🎉",
        "adv_surrogate_range": "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝕄𝔸𝕋𝐇 𠀋𠀌𠀍",
        "adv_special_token_text": "<|im_start|><|im_end|><|endoftext|><|vision_start|><|image_pad|>",
        "adv_special_like": "<|not_a_real_token|> <| im_start |> <|im_start|",
        "adv_repeated": "ab"*500,
        "adv_long_word": "x"*300,
        "adv_mixed_script_run": "hello你好こんにちは안녕Привет مرحبا नमस्ते hello",
        "adv_code_dense": "int main(){for(int i=0;i<10;++i){printf(\"%d\\n\",i);}return 0;}",
        "adv_json": json.dumps({"a":[1,2,{"b":"é你好"}],"c":None,"d":True}, ensure_ascii=False),
        "adv_markdown": "# H1\n\n- item `code` **bold** _em_\n\n```py\nx=1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n",
        "adv_empty_ish": " ",
        "adv_tabs": "\t\t\tindented\t\ttabs\t",
    }
    segs += list(adv.items())
    return segs

# ---------------------------------------------------------------- chat
def build_chat_fixtures():
    """30 fixtures across the axes the directive names."""
    tools = [
        {"type": "function", "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {"type": "object",
                           "properties": {"location": {"type": "string",
                                                       "description": "City name"},
                                          "unit": {"type": "string",
                                                   "enum": ["c", "f"]}},
                           "required": ["location"]}}},
        {"type": "function", "function": {
            "name": "run_python",
            "description": "Execute a Python snippet and return stdout.",
            "parameters": {"type": "object",
                           "properties": {"code": {"type": "string"}},
                           "required": ["code"]}}},
    ]
    U  = lambda s: {"role": "user", "content": s}
    A  = lambda s, **kw: {"role": "assistant", "content": s, **kw}
    S  = lambda s: {"role": "system", "content": s}
    # NOTE: `arguments` must be a MAPPING here, not the JSON string that OpenAI
    # clients send. The chat template iterates `tool_call.arguments|items`, and a
    # string raises "Can only get item pairs from a mapping". src/server must
    # therefore parse tool_call.function.arguments into an object before rendering.
    TC = [{"id": "call_1", "type": "function",
           "function": {"name": "get_weather",
                        "arguments": {"location": "Miami", "unit": "f"}}}]
    TOOLMSG = {"role": "tool", "content": "72F, clear", "tool_call_id": "call_1"}

    single   = [U("What is 2+2?")]
    withsys  = [S("You are a terse assistant."), U("What is 2+2?")]
    multi    = [U("Hi"), A("Hello!"), U("What is 2+2?")]
    thinking = [U("Hi"), A("Hello!", reasoning_content="The user greeted me. Greet back."),
                U("What is 2+2?")]
    toolturn = [U("Weather in Miami?"), A("", tool_calls=TC), TOOLMSG, U("And tomorrow?")]
    thinktool= [U("Weather in Miami?"),
                A("", reasoning_content="I should call get_weather.", tool_calls=TC),
                TOOLMSG]
    cjk      = [U("请用中文解释一下什么是显存带宽？")]
    codeq    = [U("Write a CUDA kernel that does a warp-level sum reduction.")]

    F = []
    def add(name, messages, **kw):
        F.append({"name": name, "messages": messages, "kwargs": kw})

    add("single_default", single)
    add("single_gen_prompt", single, add_generation_prompt=True)
    add("single_think_on", single, add_generation_prompt=True, enable_thinking=True)
    add("single_think_off", single, add_generation_prompt=True, enable_thinking=False)
    add("effort_xhigh", single, add_generation_prompt=True, reasoning_effort="xhigh")
    add("effort_medium", single, add_generation_prompt=True, reasoning_effort="medium")
    add("effort_low", single, add_generation_prompt=True, reasoning_effort="low")
    add("effort_low_think_off", single, add_generation_prompt=True,
        reasoning_effort="low", enable_thinking=False)
    add("system_default", withsys, add_generation_prompt=True)
    add("system_think_off", withsys, add_generation_prompt=True, enable_thinking=False)
    add("system_effort_low", withsys, add_generation_prompt=True, reasoning_effort="low")
    add("multiturn_default", multi, add_generation_prompt=True)
    add("multiturn_think_off", multi, add_generation_prompt=True, enable_thinking=False)
    add("thinking_preserve_default", thinking, add_generation_prompt=True)
    add("thinking_preserve_true", thinking, add_generation_prompt=True, preserve_thinking=True)
    add("thinking_preserve_false", thinking, add_generation_prompt=True, preserve_thinking=False)
    add("thinking_preserve_false_effort_low", thinking, add_generation_prompt=True,
        preserve_thinking=False, reasoning_effort="low")
    add("thinking_no_gen_prompt", thinking)
    add("tools_single", single, tools=tools, add_generation_prompt=True)
    add("tools_system", withsys, tools=tools, add_generation_prompt=True)
    add("tools_think_off", single, tools=tools, add_generation_prompt=True, enable_thinking=False)
    add("tools_effort_low", single, tools=tools, add_generation_prompt=True, reasoning_effort="low")
    add("tools_effort_medium", single, tools=tools, add_generation_prompt=True, reasoning_effort="medium")
    add("toolcall_roundtrip", toolturn, tools=tools, add_generation_prompt=True)
    add("toolcall_no_tools_decl", toolturn, add_generation_prompt=True)
    add("toolcall_with_thinking", thinktool, tools=tools, add_generation_prompt=True)
    add("toolcall_thinking_preserve_false", thinktool, tools=tools,
        add_generation_prompt=True, preserve_thinking=False)
    add("one_tool_only", single, tools=tools[:1], add_generation_prompt=True)
    add("cjk_prompt", cjk, add_generation_prompt=True)
    add("code_prompt", codeq, add_generation_prompt=True, reasoning_effort="low")
    return F

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default="tests/fixtures")
    ap.add_argument("--prior-art", default="")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.model_dir)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs(os.path.join(a.out, "tokenizer"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "chat"), exist_ok=True)

    # ---- tokenizer corpus ----
    segs = build_corpus(tk, repo_root, a.prior_art)
    total = 0
    bad_rt = 0
    path = os.path.join(a.out, "tokenizer", "corpus.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for name, text in segs:
            ids = tk.encode(text, add_special_tokens=False)
            dec = tk.decode(ids)
            total += len(ids)
            # decode(encode(x)) != x is EXPECTED where NFC normalization applies;
            # record the decoded form so the C++ gate compares against reality,
            # not against an assumption.
            if dec != text: bad_rt += 1
            f.write(json.dumps({"name": name, "text": text, "ids": ids,
                                "decoded": dec,
                                "nfc_stable": dec == text}, ensure_ascii=False) + "\n")
    print(f"tokenizer corpus: {len(segs)} segments, {total:,} tokens -> {path}")
    print(f"  segments where decode(encode(x)) != x (NFC normalization): {bad_rt}")
    if total < 100_000:
        print(f"  WARNING: {total:,} tokens is below the 100k gate requirement")

    # ---- special-token id table (the C++ side must agree exactly) ----
    specials = {}
    for t in tk.all_special_tokens:
        specials[t] = tk.convert_tokens_to_ids(t)
    added = {}
    for tok, tid in sorted(tk.get_added_vocab().items(), key=lambda kv: kv[1]):
        added[tok] = tid
    meta = {"vocab_size": tk.vocab_size, "len_tokenizer": len(tk),
            "model_max_length": tk.model_max_length,
            "eos_token": tk.eos_token, "eos_token_id": tk.eos_token_id,
            "pad_token": tk.pad_token, "pad_token_id": tk.pad_token_id,
            "bos_token": tk.bos_token, "bos_token_id": tk.bos_token_id,
            "special_tokens": specials, "added_vocab": added,
            "total_corpus_tokens": total, "segments": len(segs)}
    mp = os.path.join(a.out, "tokenizer", "meta.json")
    json.dump(meta, open(mp, "w"), indent=2, ensure_ascii=False)
    print(f"tokenizer meta -> {mp}  (added vocab: {len(added)})")

    # ---- chat template fixtures ----
    fx = build_chat_fixtures()
    cp = os.path.join(a.out, "chat", "fixtures.jsonl")
    nerr = 0
    with open(cp, "w", encoding="utf-8") as f:
        for spec in fx:
            try:
                render = tk.apply_chat_template(spec["messages"], tokenize=False,
                                                **spec["kwargs"])
                # Encode the rendered string rather than asking for tokenize=True:
                # transformers 5.x returns a BatchEncoding, and encoding the render
                # is exactly what the C++ path does (template -> string -> BPE).
                ids = tk.encode(render, add_special_tokens=False)
                f.write(json.dumps({"name": spec["name"], "messages": spec["messages"],
                                    "kwargs": spec["kwargs"], "render": render,
                                    "ids": ids}, ensure_ascii=False) + "\n")
            except Exception as e:
                nerr += 1
                print(f"  ERROR rendering {spec['name']}: {e}")
    print(f"chat fixtures: {len(fx)-nerr}/{len(fx)} rendered -> {cp}")
    return 1 if (nerr or total < 100_000) else 0

if __name__ == "__main__":
    sys.exit(main())
