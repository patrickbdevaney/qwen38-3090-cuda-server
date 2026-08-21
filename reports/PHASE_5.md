# PHASE 5 — End-to-end forward

Status: **The model runs.** 64 layers, 48 GatedDeltaNet + 16 gated attention,
FP8 KV, INT8 embedding, chunked prefill, greedy decode, CUDA graphs, VRAM budget
enforced at startup.

| | measured |
|---|---|
| decode, 4K ctx, eager | 44.3 tok/s |
| decode, 4K ctx, CUDA graph | **47.3 tok/s** |
| llama.cpp, same box, same prompt style | 38.41 tok/s |
| teacher-forced top-1 vs HF | 98.44% (2 mismatches in 128, one an exact bf16 tie) |
| worst mean KL vs HF | 7.81e-4 |

---

## 1. Four bugs, and the second one is the lesson

### The NaN
`mrun` — the running softmax max — was initialised with `memset(0xFF)`. On a
float buffer that is `0xFFFFFFFF`, which is **NaN, not `-FLT_MAX`**. `fmaxf`
quietly drops a NaN operand, but the `(mprev == -FLT_MAX)` first-tile guard does
not fire, so every running max became NaN and every prefill output with it.

### The gate that could not fail
**`std::max(x, NaN)` returns `x`.** So an all-NaN tensor measured as a **zero**
difference and *passed*. `gate_attn` had been printing

```
T128   prefill attn   0.000e+00   0.00e+00   ok
```

for a tensor that was entirely NaN. Every comparison helper now flags non-finite
values explicitly and reports `NONFINITE` rather than `ok`.

This is the most valuable thing found in this phase. **A gate that cannot fail is
worse than no gate at all, because it is believed.** The Phase 4 report claimed
prefill attention passed; it did not, and the claim has been corrected.

### The silent cuBLAS no-op
`cublasGemmEx` with a **bf16 A and an fp32 B** returns `NOT_SUPPORTED` and does
nothing. The call was unchecked, so the PV product silently produced zeros. All
cuBLAS calls are now status-checked and abort, and the attention probabilities
are materialised in bf16 so both operands share a type.

### The tile stride
The prefill score buffer is strided by the tile **capacity** (128 queries × 2048
keys), but the GEMM's `ldc` and the softmax's row stride both used the **live**
tile width. Those coincide only on a full tile, so every prompt shorter than 2048
tokens indexed the wrong rows. `T128` passed exactly while `T1` failed by 100×.

---

## 2. lm_head precision — the directive's "quantize it, gate on KL" item

| lm_head | mean KL vs HF | teacher-forced top-1 | size |
|---|---|---|---|
| BF16 | 1.47e-3 | 97.66% | 2.368 GiB |
| **INT8 g128** | **7.81e-4** | **98.44%** | **1.203 GiB** |
| INT4 g128 | 1.82e-2 | 96.88% | 0.620 GiB |
| INT4 g32 | 1.10e-2 | 99.22% | 0.790 GiB |

**INT4 costs an order of magnitude in KL and does not ship.** That invalidates the
directive's budget assumption of a 0.615 GiB `lm_head`; the real figure is
**1.203 GiB**, and §4 re-derives the context that fits.

**INT8 is not a compromise — it beat BF16.** That looks wrong and is not:
group-wise INT8 quantizes relative to the group's maximum, so for 128 weights of
similar magnitude it carries more usable precision than bf16's 8 mantissa bits
spread across each value's own exponent. The reference is itself bf16, so the
noise can fall either way.

**Clipping made INT4 worse, not better** (1.82e-2 → 5.89e-2 at g128). Searching a
clip ratio to minimise squared error is the standard trick and it backfires here:
for `lm_head` a group's outliers **are** the signal for rare tokens, so clipping
trades a small average error for a large error exactly where it matters. Recorded
in the code so it is not re-attempted.

---

## 3. CUDA graphs

A decode step issues **1750 kernels**. Captured once and replayed: **44.3 → 47.3
tok/s** at 4K.

Making one graph serve every position required moving the token id, the position
and the context length into **device memory**, and **fixing the attention split
count** so the grid shape never changes.

Fixing the split count had a second, more important effect: it made the eager and
graph paths **bitwise identical** — `gate_graph` reports 0/24 tokens and
**0/248,320 logit bits** differing. Before that they disagreed on 152,716 of
248,320 logit bits, purely because a context-dependent split count changes the
fp32 summation order of the softmax combine. **This is a prerequisite for Phase
7's losslessness test**, which compares speculation on against speculation off;
if the two decode paths already differed, that test would measure nothing.

Graphs are **bucketed by context** (≤8K, ≤32K, ≤128K). One graph sized for 128K
costs 164 splits × 24 heads × 256 dims × 16 layers = 64 MB of partial-buffer
traffic per token, which measured as 47.3 → 44.8 tok/s at 4K.

---

## 4. The VRAM budget, corrected

The directive's projection assumed a 0.615 GiB INT4 `lm_head`. §2 rules that out.

| component | directive | measured |
|---|---|---|
| LM body INT4 g128 | ~13.5 GB | 11.86 GiB |
| lm_head | 0.615 GiB (INT4) | **1.203 GiB (INT8)** |
| embed_tokens INT8 | 1.3 GB | 1.185 GiB |
| KV @ 128K FP8 | 4.0 GiB | 4.0 GiB |

Without the drafter, 128K context loads with 2.1 GiB free. **With** an INT8
drafter (1.79 GiB) it does not — which promotes the drafter-precision question
from "affects acceptance rate" to "decides whether 128K ships". §6.

---

## 5. On token exactness

The directive asks for a token-exact match over 256 greedy tokens.
`PHASE_3.md` predicted in advance this may not be reachable, because the GDN
delta rule computes `(v_t − g·Sᵀk)` — a residual — so bf16 rounding differences
against PyTorch's kernels are amplified rather than damped.

Measured: **free-running greedy matches for 2 of 4 prompts**; teacher-forced
top-1 is **98.44%** (126/128). Of the two mismatches, one has a logit gap of
**exactly 0.0** (an exact bf16 tie — a coin flip, not an error) and the other is
1.5 bf16 ulps.

So the gate reports **teacher-forced top-1 plus KL**, and says so, rather than
loosening a tolerance and calling it exact. The sample is small — 128 positions,
2 events — and a larger reference is being generated to tighten it.

---

## 6. Open items

1. **G1 is at the line.** 47.3 tok/s at 4K with an 8K-max-context graph; 44.8
   with a 128K one. The revised minimum is 45. Bucketed graphs fix this and are
   implemented; the number needs re-measuring.
2. **G2 measures 84%** against an 85% bar (37.5 tok/s at 64K vs 44.8 at 4K).
   Phase 4 §6.1 identifies the fix: the 6 warp reductions per KV position cost
   ~2 ms at 128K, and giving each lane 16 dims would cut shuffle rounds from 30
   to 12.
3. **G7 measures 21.5 GiB peak (23.1 GB)** against a 22.5 GB bar, and ~0.5 GiB of
   that is the desktop. The 256 MB prefill GEMM workspace and the attention
   partial buffers are the obvious places to trim.
4. The **larger reference** (10 prompts × 96 tokens) needs regenerating; the
   first attempt OOMed because it shared the GPU with a benchmark.

## The single riskiest open question

**Sharpened by §2 and §4, and now quantitative.** `lm_head` must be INT8, costing
0.583 GiB more than the directive assumed. The remaining budget at 128K is
2.1 GiB, and an INT8 DFlash2 drafter needs 1.79 GiB of it — leaving 0.3 GiB,
which is inside the safety margin.

So the drafter precision no longer just trades acceptance rate against memory: at
INT8 it leaves **no room at 128K**, at INT4 (0.95 GiB) it does. And
`gemma-cuda-hybrid` measured that quantizing exactly this class of drafter
collapsed acceptance from 13.33 to 11.14 — *"the bf16 draft IS the moat"*.

**The three-way trade is now explicit: 128K context, drafter precision, and
acceptance length cannot all be maximised.** Phase 7 measures the acceptance cost
of INT8 and INT4 drafters first, before any speedup number is quoted.
