# PRIOR_ART.md — harvest catalogue

Phase 0, read-only. Nine repos catalogued: eight of the user's, plus the official
`z-lab/dflash` reference implementation.

**The load-bearing warning from the directive is confirmed and then some.** Every one of the
user's CUDA repos targets Jetson Thor (sm_110a Blackwell) with NVFP4 weights. But the grep came
back *better* than feared: **`wgmma`, `TMA`, and `cp.async.bulk` appear in exactly one file
across all repos, and that file is a capability probe, not a kernel**
(`deepseek-v4-flash-0731-cuda/tools/cap_probe.cu`). One `sm_100` (`kernels/cutlass_moe.cu`).
Zero `sm_90`.

The real contamination is narrower and more specific:

1. **NVFP4 weight containers** — a *format*, not an instruction. Its dequant uses
   `__nv_cvt_fp4x2_*` hardware intrinsics that sm_86 lacks. laguna's `README` records that
   replacing a LUT with the HW converter was worth 2.5x; on sm_86 we revert that trade.
2. **FP8 tensor-core MMA** (`tc_fp8_gemm.cu`, `tc_moe_gemm.cu`, `tc_verify_gemm.cu`) — sm_86
   has no FP8 MMA at all. These must become BF16 `mma.sync.m16n8k16`.
3. **`__nv_fp8` / e4m3 conversions** — available in hardware sm_89+, **software on sm_86**.
   They still work, they just cost ALU. Budget it in the FP8 KV path.
4. **`-arch=sm_110a` build flags** in ~40 shell scripts.

Everything else — the architecture, the gate ladder, the methodology, the serving layer, the
host-side C++ — transfers.

---

## 1. Destination map

| New module | Primary source | Secondary | Verdict |
|---|---|---|---|
| `src/config/` | laguna `include/laguna_config.h` | — | rewrite. Keep the **"no defaults, missing key = hard error"** policy and the typed per-layer-type `RopeSpec`. |
| `src/loader/safetensors.cpp` | laguna / deepseek / gemma `include/safetensors.h` (three near-identical copies) | — | port as-is; drop the FP4 dtype branch |
| `src/loader/` (arena) | laguna `include/laguna_weights.h` | deepseek `include/weight_store.h` | rewrite. Single-arena streaming repack, no host double copy. Keep deepseek's `posix_fadvise(DONTNEED)` trick. **Drop all unified-memory logic** — discrete VRAM is a different machine. |
| `src/tokenizer/bpe.cpp` | laguna `include/tokenizer.h` + `unicode_tables.h` | deepseek `tokenizer_dsv4.h` | rewrite. The hand-rolled `\p{L}`/`\p{N}` matcher and 2-stage pretokenizer are the expensive, model-agnostic part — `std::regex` cannot do this. |
| `src/tokenizer/chat_template.cpp` | gemma `include/tokenizer.h::chat_prompt` | — | reference only; Qwen3.8 grammar is new |
| `src/kernels/gemv_w4a16.cu` | laguna `kernels/gemm.cu` | — | **rewrite, but this is the most portable asset in any repo**: a pure CUDA-core, bandwidth-bound GEMV with `__ldcs` evict-first, `uint4` 16-byte loads, offline repack to fragment order. No `mma.sync` anywhere in it. |
| `src/kernels/gemm_w4a16.cu` | — | gemma `kernels/tc_verify_gemm.cu` (fragment layout only) | net-new. Marlin-style `cp.async` + `mma.sync.m16n8k16`. |
| `src/kernels/attn_prefill.cu`, `attn_decode.cu` | laguna `kernels/attention.cu` | deepseek `mla_attn.cu::sparse_attn_kernel_s` | rewrite. laguna is `#define HD 128`; we need **256**. Its `red_acc[NW][G][HD]` at G=6, HD=256 = 24 KB — fits. **Do not port deepseek's `kernels/attention.cu` prefill SDPA: its smem is O(seq)** and would cap us near 25K tokens. |
| `src/kernels/kv_fp8.cu` | **deepseek `include/kv_pack.h`** | — | **the single highest-value file found.** FP8 e4m3 + UE8M0 packed KV row with a `memcmp`-exact round-trip test (not a tolerance test). Two arguments to keep verbatim: leave the RoPE dims full precision, and never store fp32 scales. |
| `src/kernels/elementwise` (rmsnorm/rope/swiglu) | laguna `kernels/elementwise.cu` | — | port as-is modulo shapes |
| `src/kernels/sampling.cu` | gemma `forward.cu::k_argmax/k_sample` | deepseek `include/topk_radix.h` | rewrite. Take Gumbel-max for lossless spec sampling and topk_radix's **exact tie-break** (lowest index wins, `T` sentinel not `-1`) — downstream sums selected rows in order and fp32 is non-associative. **Fix the inherited bug**: gemma's samplers operate on uncapped logits and diverge from HF temperature semantics. |
| `src/model/forward.cu` | laguna `src/forward.cu` (Engine/Session split, graph capture) | deepseek `include/dscratch.h` (arena w/ high-water), `src/engine.cu` | rewrite. Adopt deepseek's **"exactly three KV mutators"** invariant (`prefill_full`, `extend`, `rewind_to`) and its unified chunked path: prefill, prefix-extend and spec-verify are all callers of one batched-forward primitive. |
| `src/spec/dflash_draft.cu` | **`z-lab/dflash` `dflash/model.py:478-597`** (math) | dspark `src/draft.cu` (CUDA kernels, device argmax, incremental context) | net-new. See §3. |
| `src/spec/verify.cu` | **`z-lab/dflash` `model.py:289-292`** (greedy), `:94-124` (sampling) | dspark `src/decode.cu` | net-new |
| `src/spec/state_rollback.cu` | **`z-lab/dflash` `model_mlx.py:539-653`** (`_GDNStateCapture`) | caddtree `ddtree_state.py` | net-new. See §3.3 — this validates the directive's design (A). |
| `src/spec/ksweep.cpp` | caddtree `no_train_suite/profiles/thor/c_verify_linear.json` (schema) | `dflash/benchmark.py:188-216` (metric defs) | port schema |
| `src/cache/radix_prefix_cache.cpp` | gemma `forward.cu::engine_prefill_cached` | deepseek `rewind_to` | rewrite. gemma's is a **single-slot LCP** cache; the valuable part is its argument that *LCP reuse is bit-exact by construction*. Single-slot thrashes under interleaving (their own WIKI §10). |
| `src/server/http.cpp` | bonsai `server/bonsai-server.cpp:1184-1266` | gemma `forward.cu::run_server` | port with rewrite |
| `src/server/openai_api.cpp` | bonsai `:901-944, 1269-1488` | — | port with rewrite; see §2 for the gaps |
| `src/server/reasoning_parser.cpp` | **bonsai `:77-106` `ReasoningSplitter`** | deepseek `include/stream_parse.h` | port nearly verbatim. Fix `HOLD` (should be `max(len(open),len(close))-1`), parameterize tags, add `preserve_thinking`. |
| `src/server/metrics.cpp` | bonsai `:109-150` `Metrics` + `dspark.h:110-128` `DSparkStats` | — | port the counter set; **re-render as Prometheus** (bonsai emits JSON) and make `errors` atomic (current code races). |
| `src/clients/tui.cpp` | bonsai `server/chat.cpp` (140 L) | laguna `tools/lgchat.cc` | port as-is |
| `src/clients/webui/` | bonsai `server/webui.html` (211 L) | gemma `include/webui.h` | port as-is |
| `third_party/` | gemma / laguna / bonsai vendored `httplib.h` + `json.hpp` | — | port as-is |
| `bench/` | laguna `tests/bench_decode.cu`, `tools/bench_server.sh`, `k_sweep.sh` | deepseek `tools/decode_model.py`, `dprof` | port. See §4. |
| `tools/dump_reference.py` | deepseek `ref/gen_units.py` + per-primitive golden `.safetensors` | gemma `scripts/ref_forward.py` | rewrite. deepseek's per-primitive golden factoring beats gemma's raw `.bin` dumps. |

---

## 2. The serving contract (from `local-agent-bootstrap`)

`local-agent-bootstrap` is the acceptance test, and it demands three things **not on the
directive's feature list**. Missing any of them silently breaks the harness:

1. **`GET /health`** — a llama.cpp endpoint, not OpenAI. Every lifecycle command polls it.
   Only needs HTTP 2xx; body unread.
2. **A `timings` object on non-streaming chat completions** — `{predicted_per_second,
   prompt_per_second, predicted_n}`. `bin/agent:433-441` **hard-exits** without it, which
   breaks `agent bench`, `effective_mode()`, `agent code` and `agent status`.
3. **A configurable model id** (`--alias`) that equals `/v1/models` `data[0].id`, because the
   OpenCode config key must match it. A hardcoded id will not do.

Plus: the server must **accept a request with `model` absent** (`toolcall-battery.py` omits it),
must **not require auth** but must accept an arbitrary bearer token (`local-no-auth`), and must
send the first SSE byte within 120 s of a long prefill (`headerTimeout`) — the role-prelude
chunk covers this.

Tool calling is **modern OpenAI `tools`/`tool_calls` only**; no legacy `functions`. `arguments`
is a JSON string. `--sweep` pads to 40+ decoy tools, so tool serialization must be byte-stable
at a fixed prompt position or prefix caching collapses (their `docs/STATUS.md:50`: a mid-session
tool-list change reprocessed 8,845 tokens).

**Phase 8 acceptance test is `projects/run-all.sh`**: five multi-turn tool-using coding tasks
with objective verifiers that also check the agent did not cheat (e.g. that the test file is
byte-identical to seed while the source changed). Baseline to beat: 5/5 projects, 38/38 checks.

**Gap list — nothing in any repo implements these:** tool calling (bonsai has *zero*),
`reasoning_effort`, `stream_options.include_usage`, Prometheus `/metrics`, `preserve_thinking`.

---

## 3. DFlash2 — the spec, derived from `z-lab/dflash`

The official repo is small and normative. **`dflash/model_mlx.py` is the file that matters**:
the README states the MLX backend is the one supporting DFlash2 for Qwen3.8-27B, and it is the
only place recurrent-state rollback exists. `dflash/model.py` is the clean algorithmic twin.

### 3.1 Draft forward
- Block is `[last_accepted_token, MASK x 7]`, `mask_token_id = 248070`, embedded with the
  **target's** `embed_tokens` scaled by `input_embedding_scale`.
- The 5 taps are the residual stream **after** decoder layers 5, 19, 33, 47, 61
  (`offset=1` into HF's `output_hidden_states`), concatenated **in that order** →
  `[T, 25600]` → `fc` → `hidden_norm` (RMSNorm) → one context vector `[T, 5120]`.
- **That vector is the K/V source for every draft layer** — it is not a residual stream. Each
  layer re-projects the same `target_hidden` with its own `k_proj`/`v_proj`, concatenated with
  the block's own K/V.
- `is_causal=false` + `sliding_window=2048` → a **symmetric band mask**. The 8 block positions
  attend bidirectionally to each other. That is what lets one forward denoise the block.
- **Two-tap dynamic conv, decoded**: `base_kernel[2, 2, 5120]` — the leading 2 is
  *prepare vs finish* (pre- and post-sublayer), not the taps. Per output channel `c` in group
  `g = c/16`, at position `t`:
  `y[t,c] = (base[0][c] + dyn[t,0,g])*x[t,c] + (base[1][c] + dyn[t,1,g])*x[t-1,c]`,
  with `x[-1] = 0`. `conv_group_size=16` is the rank-reduction on the *input-dependent* part
  (1280 dynamic scalars, 320 groups) while the static part is per-channel.
  **This tap is causal even though attention is not.**
- **Logits come from the TARGET's `lm_head`** applied to draft hidden states — confirmed in
  code, consistent with the drafter having no head of its own.

### 3.2 Candidate selector
A **first-order Markov chain over the top-16 lattice**, decoded greedily left to right (no
Viterbi, no beam). With `pred = anchor`, for each of 7 positions:
```
score[k] = unary[i,k] + <predecessor_codebook[pred] (*) hidden_projection@h[i], successor_codebook[cand[i,k]]>
```
greedy: `sel = argmax_k score`; sampling: `q = softmax(score/T)` over the 16, sample, **keep the
16-wide q row** (needed for the lossless rule). ~7 x 16 x 256 MACs — trivial. The codebooks are
`248320 x 256` each but only 17 rows are gathered per position.

### 3.3 Acceptance — **linear chain, no tree, no tau**
- **Greedy**: longest matching prefix of exact argmax equality, `cumprod` to stop at first
  mismatch, bonus = target argmax at the first rejected slot. Nothing else.
- **Sampling**: standard Leviathan/Chen rejection sampling, `u*q < p`, with the DFlash2 twist
  that `q` is **sparse over 16 candidates** — recover `q_i` by masked sum, form the residual by
  `scatter_add_(-q)` onto only those 16 slots of full-vocab `p`, clamp, renormalize.
- **There is no `tau` in the official repo.** `grep -rn "tau|threshold|epsilon|lossy"` returns
  zero. "τ" in this ecosystem is the *metric* mean-accepted-length. The threshold knob
  (`DFLASH_ACCEPT_EPS`, default 0) is a **third-party lossy addition** in the caddtree repo,
  hard-gated to T>0. Per the directive §9: off by default, documented as lossy, never called
  "DFlash2".
- **Tree verification: do not build it.** caddtree implemented CaDDTree, verified it correct on
  a GDN hybrid, optimized it 3.6x, and their own honest conclusion is that it **does not beat
  strong-draft linear DFlash** — "depth beats breadth for a strong draft". DFlash2's selector
  already does the coherent-path job the tree approximated.

### 3.4 GDN state rollback — **the reference solves it, and it validates design (A)**
`model_mlx.py:539-653`, `_GDNStateCapture`. Not a tensor snapshot — a **capture-and-replay**:
1. Capture, per GDN layer, the recurrence inputs `(q,k,v,a,b,A_log,dt_bias,S_seed,mask)` for the
   verify block, plus the conv input window `[conv_state || qkv]`. Note `S_seed` is the state
   *before* the block.
2. Run the verify forward normally; the state is allowed to advance wrongly.
3. On partial acceptance, **replay the recurrence from `S_seed` over only the first
   `accepted+1` positions** and overwrite the cache.
4. The **short-conv state is restored by slicing** the captured `conv_input` at
   `[accepted+1 : accepted+K]` — no recompute.
5. Full-attention layers take the ordinary `trim()` path in the same loop.

This is the directive's design (A) with two refinements: the conv state is *sliced, not
replayed*, and the seed state is the only per-layer buffer needed (144 MiB fp32 for 48 layers).

caddtree's `no_train_suite/designs/gdn_rollback.md` specifies the debug gate we should
implement and they did not: for the first N rounds, recompute each layer's state from scratch
over only the accepted tokens from the seed and `assert allclose`. Their own audit rates
rollback correctness as *"needs runtime verification"*. **We close that.**

### 3.5 A free long-context win
All 5 draft layers are `sliding_attention` with window 2048, so the reference caps the
target-hidden context at `sliding_window - 1 = 2047` positions (`model_mlx.py:757-761`).
**The draft KV cache and the context projection are therefore constant in sequence length.**
At 128K context this is the difference between a drafter that scales and one that does not.

### 3.6 What is genuinely net-new
Nothing in any repo implements DFlash **2** in CUDA. dspark's `src/draft.cu` is DFlash **1** —
no dynamic conv, no candidate selector. And no repo does GDN rollback in CUDA; the only real
implementation is the MLX one above.

---

## 4. Methodology to carry forward

Ranked by transfer value. Sources: `gemma-cuda-hybrid/CUDA_ENGINEERING_CONSTITUTION.md`,
`AGENTIC_OPTIMIZATION_METHODOLOGY.md`, `laguna/DIRECTIVE.md` §10.

1. **Correctness gates before speed gates, always. A faster wrong kernel is worth zero.**
   If a gate fails, stop; do not build on an unverified layer.
2. **Back-to-back A/B, median of 5, never compare numbers taken at different times.** Thermal
   drift is real and measured: gemma saw identical code swing 94↔108 tok/s. Our own microbench
   already shows the 3090 boosting 1710→1965 MHz.
3. **Precision unification (W4A16 everywhere) is what makes bit-exactness possible at all.**
4. **The draft must stay high precision.** gemma tested draft→FP4 and draft→FP8: both collapsed
   acceptance 13.33→11.14. "The bf16 draft IS the moat." Verify guarantees output, so draft
   numerics cost only τ — but τ is the multiplier. **This directly challenges our plan to run
   the drafter at INT8; §5 of PHASE_0.md treats it as the open question it is.**
5. **Re-test dismissed levers after the kernel they'd use improves.** gemma's key meta-lesson:
   the TC kernel was neutral for the lm_head until it had repack + prefetch + 16B loads, then
   routing lm_head through it was +9.7%.
6. **Keep a won/lost/neutral ledger with reasons**, and do not re-litigate it.
7. **Maintain a workload basket, not one prompt.** Acceptance on code is roughly *double*
   acceptance on prose, so a single number is not a useful summary of a speculative server.
   gemma's own §10 admits violating this and reporting a one-prompt number.
8. **Repack weights offline into fragment order, cached, done at warm-up** so graph capture
   never sees a `cudaMalloc`.
9. **Alignment gotcha**: per-layer weight pointers out of safetensors are **not 16-byte
   aligned**, so `uint4` loads crash. Only separately allocated, repacked buffers are safe.

### Gate ladder
laguna's is the template and we adopt it (see PHASE_0.md §7), with three amendments its own
history argues for:

- **Add gate B1c: batched-forward ≡ sequential-decode, bit-exact, on uniform-random token ids.**
  laguna added this late and it caught two bugs that would have made every acceptance number
  meaningless. The invariant it enforces: **the attention split count must not depend on M**,
  or decode and verify round differently and 1 ulp at layer 0 amplifies ~1.4x/layer into a
  different argmax.
- **Split the kernel gate into G3-GDN and G3-Full**, and add a gate that the 3:1 layer
  interleaving is right — a wrong layer map passes every per-kernel gate.
- **Run capability/bandwidth probes before the roofline.** Done; see `bench/microbench.cu`.

### Two honest cautions about the prior numbers
- gemma's headline "118 tok/s vs vLLM 107.5" is **one prompt, 80 tokens, short context**. The
  only controlled long-workload comparison in the repo showed **58 vs 100** and was never redone.
- **Neither laguna nor deepseek ever ran llama.cpp or vLLM locally.** Both closed their
  head-to-head gate against *published vendor numbers*, and laguna's `BENCHMARK_COMPARISON.md`
  does not exist. Per directive §5.3, we measure both ourselves. `reports/BASELINES.md` is
  therefore genuinely new work, not a port.

### Bugs inherited if you copy carelessly
- `gemma/scripts/{gate_self,bench}.sh` both `cd ~/gemma-cuda-server` (the *sibling* repo), so
  gate passes claimed by running them from `gemma-cuda-hybrid` are suspect.
- bonsai `ReasoningSplitter` `HOLD=8` is `strlen("</think>")`, not `-1`.
- bonsai `g_metrics.errors++` is unsynchronized on a non-atomic `uint64_t`.
- laguna `include/suffix.h` documents a drafter that must index up to but **not including** the
  last position, or the current suffix matches itself and acceptance silently goes to zero.
