# PHASE 1 — Reference harness and loader

Status: **GATE PASSED.** Five gates green under `ctest`. No kernels written yet;
Phase 2 is the first CUDA work.

```
1/5 nfc ............... Passed    8,612 NFC cases, 0 mismatches
2/5 tokenizer ......... Passed    232 segments / 393,815 tokens, 0 failures
3/5 chat_template ..... Passed    30/30 fixtures byte-exact
4/5 config ............ Passed    27 shape assertions
5/5 dequant ........... Passed    229,376 values, BIT-EXACT
```

The directive's Phase 1 gate is *"tokenizer round-trips 100k tokens of mixed
code/prose/CJK exactly; chat template output matches HF `apply_chat_template`
byte for byte across 30 fixtures covering thinking on/off, `preserve_thinking`
on/off, tool calls, multi-turn, and system prompts."* Both halves pass, at
3.9x the required corpus size.

---

## 1. What was built

| Module | File | Gate |
|---|---|---|
| Unicode / NFC | `src/tokenizer/unicode.h` + generated `unicode_tables.h` | `gate_nfc` |
| Byte-level BPE | `src/tokenizer/bpe.{h,cpp}` | `gate_tokenizer` |
| Chat template | `src/tokenizer/chat_template.{h,cpp}` | `gate_chat_template` |
| Config → ModelShape | `src/config/model_shape.{h,cpp}` | `gate_config` |
| safetensors mmap | `src/loader/safetensors.{h,cpp}` | exercised by `gate_dequant` |
| W4A16 unpack | `src/loader/w4a16_unpack.h` | `gate_dequant` |
| Oracles | `tools/dump_{tokenizer_ref,dequant_ref,reference}.py`, `tools/gen_{unicode_tables,nfc_cases}.py` | — |

Dependencies added: `nlohmann/json` 3.12.0 and `cpp-httplib` 0.53.1, both
vendored header-only, as the directive allows. Nothing else.

---

## 2. The finding: transformers silently rewrites the pretokenizer

**`transformers` 5.15.1 replaces the pre_tokenizer pattern that ships in
`tokenizer.json` with a legacy Qwen2 one that drops `\p{M}`.**

```
on disk       [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ ... [^\s\p{L}\p{M}\p{N}]+ ...
transformers  [^\r\n\p{L}\p{N}]?\p{L}+        ... [^\s\p{L}\p{N}]+      ...
```

`tokenizers.Tokenizer.from_file()` keeps the on-disk pattern; `AutoTokenizer`
does not. They disagree on every script with combining marks.

Measured on this checkpoint:

| | |
|---|---|
| vocab entries containing a combining Mark | **6,354** |
| of those, entries `AutoTokenizer` can never emit | **5,504 (2.2% of the vocab)** |
| of those, entries the on-disk pattern emits as one token | ~99.9% |

Examples: Thai `ที่` (id 148285) becomes two tokens; Devanagari `मध्ये`
(id 223177, a single vocab entry) becomes four.

**A tokenizer that cannot produce 2.2% of its own model's vocabulary is the
broken one**, so the on-disk pattern is authoritative and is what
`src/tokenizer` implements. The oracle was regenerated through
`tokenizers.Tokenizer.from_file`; `AutoTokenizer` is now used only for Jinja
chat-template rendering.

This surfaced the right way round: the first gate run failed on exactly the five
Indic/Arabic/Thai segments, and investigation showed the *oracle* was wrong, not
the implementation. That is what the "numerics before speed" rule is for.

**Consequence to carry into `BENCHMARKS.md`:** this server will tokenize Indic,
Arabic and Thai text differently from — and better than — any `transformers`- or
vLLM-based server. A token-level comparison against those stacks will not match
on such text, and that is expected rather than a defect in us. It also means our
tokens-per-byte on those scripts is materially better, which flatters tok/s
comparisons; any such comparison must be stated in bytes as well as tokens.

---

## 3. Numerics pinned

### W4A16 unpack — bit-exact
`gate_dequant` checks 229,376 values across 7 tensors covering every distinct
shape class (mlp gate/down, attention q/k, GDN qkv/z/out) against
compressed-tensors' own `unpack_from_int32` + dequantize. **Zero int4
mismatches, worst |diff| exactly 0.0** — not "within tolerance", identical.

The format, read out of
`compressed_tensors/compressors/pack_quantized/helpers.py` and then verified
rather than assumed:

- a weight is int8 in `[-8, 7]`, stored as `nibble = w + 8`
- packing is **dense little-endian** along the packed dim: element `i` occupies
  global bits `[4i, 4i+4)`. Because 4 divides 32, element `e` of a row lives in
  word `e/8` at bit `4*(e%8)` and never straddles a word boundary — which is
  exactly what makes a dequant-fused-into-the-load GEMV cheap on this format,
  and is a Phase 2 input.
- `weight_packed` is packed along dim 1 (input features)
- `weight_zero_point` is packed along dim 0 (**output** features). This is the
  easy mistake: its `[out/8, G]` shape reads like a row-major `[rows, cols]`,
  but row `o` is nibble `o%8` of word `[o/8][g]`. It carries the same `-8` offset.
- dequant is `(q - zp) * scale` per (output row, group)

Doing this before any GEMV exists was deliberate. A fast kernel over a wrong
unpack is not an optimization; it is a bug that produces fluent nonsense.

### Config — no defaults
`ModelShape::from_file` throws on any missing or unexpected key. Two invariants
are checked at parse time because getting them wrong is directive failure
mode #1 (fluent output that decays at long context):

- `mrope_section` must sum to `rotary_dims / 2`
- `head_dim * partial_rotary_factor` must be an integer

All 27 Gate 0 assertions pass, and every derived constant reproduces Phase 0
exactly: 32,768 KV elements/token, 144.0 MiB fp32 GDN state, 10,240-wide GDN qkv
projection, 4.00 GiB of FP8 KV at 128K.

---

## 4. Chat-template semantics now pinned by fixtures

Each of these is easy to get silently wrong and each is covered:

- `reasoning_effort` defaults to `xhigh`; **`medium` injects nothing at all**, so
  "medium" and "no instruction" are the same prompt.
- `preserve_thinking=false` strips `<think>` only from assistant turns at or
  before `ns.last_query_index`, which is found by scanning **backward** for the
  last user turn whose trimmed content is not wrapped in `<tool_response>`.
  Tool-result turns therefore do not reset it.
- Tool results render as a **user** turn containing `<tool_response>`, and
  consecutive tool messages share a single `<|im_start|>user` block.
- `tool_calls` render as Qwen XML (`<function=NAME><parameter=K>`), **not JSON**.
- **`tool_calls[].function.arguments` must be an object.** The template does
  `arguments|items`; OpenAI clients send a JSON *string*. `src/server` must parse
  it before rendering or every tool call raises. This was found by the oracle
  failing, not by reading.
- Only an **explicit** `enable_thinking=false` pre-closes the generation prompt
  with `<think>\n\n</think>\n\n`; undefined leaves it open.
- `tools` serialise with Python `json.dumps` separators (`", "` / `": "`) in
  insertion order, so there is a small purpose-built serialiser rather than a
  call to `nlohmann::dump()`.
- Jinja's `|trim` is Python `str.strip()`, whose whitespace set is Unicode
  White_Space **plus U+001C..U+001F**. Off by one character there shifts every
  subsequent byte of the render.

Image and video content parts throw `UnsupportedContent`, which the server maps
to a 400. v1 is text-only by design; the hook is left in place, not half-built.

---

## 5. The reference oracle, and a scope note

`tools/dump_reference.py` dumps, for 4 fixed prompts: all 65 hidden states
(embedding + 64 layers), the last position's full logits plus top-1024 for every
position, the DFlash2 tap layers, and a 32-token greedy continuation. This is the
ground truth for every kernel gate in Phases 3–5.

Sanity of the reference itself: `"The capital of France is"` → `" Paris"`;
the fibonacci prompt completes to the correct recursion; `60 miles / 1.5 hours`
→ `" 40 miles per hour"`; the Chinese prompt produces coherent Chinese. The
checkpoint is sound.

**Scope note, because it would otherwise look like drift.** The oracle consumes a
BF16 checkpoint materialised from the INT4 weights
(`tools/dequantize_checkpoint.py`, 51 GB on disk). That copy is a **development
artefact only**. The server never reads it, links it, or knows it exists: it
loads the **INT4 AWQ checkpoint directly into VRAM at 13.62 GiB** and computes
W4A16, which is the entire point of the project. The BF16 copy exists for two
reasons, both external:

1. `compressed-tensors` 0.18.0 **crashes** decompressing this checkpoint —
   `unpack_from_int32` is called twice on `weight_zero_point`, so the second call
   gets `int8` where it wants `int32`. Both `run_compressed=True` and the default
   path hit it.
2. Even without that bug, in-process decompression needs ~53 GB of RAM on a 60 GB
   box that is also running a desktop.

Because the BF16 weights are exactly the dequantised INT4 weights, this is the
correct reference for the Phase 5 gate, which asks for a token-exact match
against HF *"on the same quantized weights"*. The 51 GB copy is deletable once
the fixtures are committed.

---

## 6. Plan changes

1. **Add a `--tokenizer-compat` flag, defaulting to `disk`.** Given §2, someone
   comparing us against a transformers-based server will see token-count
   differences on Indic/Arabic/Thai. A `hf-legacy` mode that reproduces the
   transformers pattern makes that comparison possible without pretending the
   legacy behaviour is correct. Cheap: one boolean in the pretokenizer.
2. **Gate the drafter's tokenizer too.** DFlash2 shares the target's embedding
   and `lm_head`, and its selector codebooks are indexed by token id, so a
   tokenizer disagreement between draft and target would silently destroy
   acceptance. Same gate, run against the drafter's config in Phase 7.
3. The CUDA AWQ repack (`src/loader/awq_repack.cu`) moves to **Phase 2**, where
   the GEMV that consumes its layout exists. Repacking into a layout before
   knowing what the kernel wants would be guesswork; the host-side unpack is
   already gated and is the reference the repack must reproduce.

---

## The single riskiest open question

Unchanged from Phase 0, and now one phase closer: **does the DFlash2 drafter
survive INT8 quantization with its acceptance length intact?**

Phase 1 added one wrinkle to it. The drafter has no `lm_head` of its own — it
runs the *target's*. So the planned `lm_head` BF16→INT4 repack, which reclaims
1.753 GiB and is load-bearing for the 128K claim, sits on the drafter's critical
path as well as the model's. A quality regression there costs output quality
**and** acceptance length, and the two would be easy to confuse. The Phase 2 KL
gate on `lm_head` must therefore be run before Phase 7, not alongside it.
