# PHASE 4 — Gated attention

Status: **GATE PASSED.** All stages match the transformers reference at 1, 8 and
128 tokens **and at positions 0, 1, 4095, 32767, 131071**.

| | measured |
|---|---|
| decode attention, 16 layers, 4K ctx | 0.90 ms/token |
| decode attention, 64K ctx | **5.26 ms/token** |
| decode attention, 128K ctx | **9.66 ms/token** (445 GB/s of KV, 49% of peak) |
| KV per token of context | 32 KiB across all 16 layers |
| FP8 cache agreement with `float8_e4m3fn` | 94.7–97.5% of bytes exact, rest one code |

---

## 1. Four things the reference does that the config does not tell you

Each of these was read out of `modeling_qwen3_5.py`, and three of the four were
caught by the gate failing rather than by reading carefully enough.

1. **`q_proj` is not 24 query heads.** It emits `num_heads * head_dim * 2 =
   12288`, which is 24 × `[query(256) | gate(256)]` **interleaved per head**. The
   gate is applied at the very end as `attn_out * sigmoid(gate)`, before
   `o_proj`. The config's `output_gate_type: "swish"` describes the *GDN* gate,
   not this one — the code says `sigmoid`.
2. **There are two different RMSNorms in this model.** `Qwen3_5RMSNorm` (used by
   `q_norm`, `k_norm`, and the layer norms) is
   `norm(x.float()) * (1.0 + weight.float())` kept in **fp32** through the weight
   multiply. `Qwen3_5RMSNormGated` (used by the GDN) rounds to **bf16 before**
   the weight and has no `1.0 +`. The weights are initialised to **zeros**, so
   the `1.0 +` is load-bearing, not cosmetic. Implementing the gated form here
   put q 96% out.
3. **RoPE covers the first 64 dims**, confirmed from
   `q[..., :rotary_dim]` / `q[..., rotary_dim:]` — not assumed. The other 192
   pass through untouched.
4. **cos/sin are cast to the activation dtype (bf16)** before the rotation, so
   the rotation itself carries only 8 mantissa bits.

## 2. The mrope bug the directive predicted, caught exactly where it said

The directive's failure mode #1 is *"output is fluent, benchmarks at 4K look
fine, quality collapses at 32K+"*, with the mitigation being position-specific
validation at 0, 1, 4095, 32767 and 131071. That is precisely what happened:

```
far_4095    q norm+rope   rel 7.99e-01   FAIL
far_32767   q norm+rope   rel 8.25e-01   FAIL
far_131071  q norm+rope   rel 9.11e-01   FAIL
T1_pos0     q norm+rope   rel 0.00e+00   ok      <- passes at position 0
```

**Cause: `__powf` for `inv_freq`.** It carries ~1e-4 relative error, and the
angle is `position × inv_freq`, so at position 131071 that is **13 radians of
angle error**. Position 0 is exact because the angle is zero. A 4K benchmark
would have looked perfect.

Secondary: `__sinf`/`__cosf` lose accuracy long before an argument of 131071
radians, also silently.

Fixed by computing `inv_freq` in **double** and rounding once to float (matching
torch's float32 `inv_freq`), taking the **product in float32** (matching
`inv_freq.float() @ position_ids.float()`), and using accurate `sinf`/`cosf`.

The residual is now **≤ one bf16 ulp** — which of two adjacent bf16 values a
float32 angle rounds to, on a table the reference itself stores in bf16. That is
a tie, and it is **200× tighter than the bug it replaced**.

**On mrope generally:** frequency `i` takes its position from axis T/H/W by
`i % 3`, which is exactly sections `[11,11,10]` over 32 frequencies. For
text-only input all three axes carry the same position, so the interleave is an
identity — but it is implemented generally rather than collapsed, because a
future vision path would silently get it wrong otherwise.

## 3. A real indexing bug, and why T=128 hid it

The prefill score buffer is strided by the tile **capacity** (128), but the
softmax/accumulate/finish kernels derived the head index as
`blockIdx.x / q_tile` while the launch enumerated only the **live** rows
(`nq × qn`). Those agree only when `qn == q_tile`. Every prefill with fewer than
128 queries — and every partial final tile — addressed head 0 for all heads.

`T128` passed exactly (0.0e+00) while `T1` failed by a factor of 100. **A gate
that only tested full tiles would have shipped this.**

## 4. Decode performance: 15.15 → 9.66 ms at 128K

KV traffic is 32 KiB per token of context, so 128K context means **4 GiB read per
decoded token** — comparable to the entire 12.4 GiB weight stream. This kernel's
efficiency directly sets long-context decode speed.

Structural decision made first: **all 6 query heads that share a KV head live in
the same block**, so K and V are read once rather than six times. At 6:1 GQA the
naive mapping would have made KV traffic 24 GiB per token.

Then measured, by stubbing pieces out rather than guessing:

| variant | 128K ms/token | KV GB/s |
|---|---|---|
| baseline | 15.15 | 284 |
| fp8 decode stubbed out | 10.17 | 422 |
| shuffle reductions stubbed out | 13.06 | 329 |

So the **e4m3→float conversion was a third of the kernel** — 16 conversions per
lane per KV position at ~7 ALU ops each, because sm_86 has no FP8 hardware.
Replaced with a **1 KB shared-memory LUT**: one load instead of seven ops. Bank
conflicts are mild because e4m3 byte values across a warp are effectively random.

Vectorising the KV loads to 8 bytes per lane changed nothing — the compiler was
already doing it. Recorded because it was a predicted win that was not one.

Split count swept over 1/2/4/8 waves; 8 is the knee.

**Result: 15.15 → 9.66 ms at 128K, 284 → 445 GB/s.**

## 5. Where this leaves the decode budget

| context | weights | GDN | attention | total | tok/s |
|---|---|---|---|---|---|
| 4K | 16.5 | 0.76 | 0.90 | 18.2 ms | ~55 |
| 64K | 16.5 | 0.76 | 5.26 | 22.5 ms | ~44 |
| 128K | 16.5 | 0.76 | 9.66 | 26.9 ms | ~37 |

**G2 (64K decode ≥ 85% of 4K) currently projects to ~81%** on kernel time alone.
It is close, and the projection is pessimistic — it omits the norms, sampling and
residuals, which are context-*independent* and therefore improve the ratio. But
it is not yet a pass, and it is called out rather than rounded up. Phase 5
measures it end to end.

## 6. Open items

1. **The 6 warp reductions per KV position** are the next ~2 ms at 128K. Giving
   each lane 16 dims would let 16 lanes cover a head and 2 heads share a warp,
   cutting shuffle rounds from 30 to 12 per position. Phase 9.
2. **445 GB/s is 49% of peak**; the remaining gap is softmax and FMA ALU.
3. **The FP8-vs-BF16 KV quality gate** — top-1 agreement ≥ 99.5% over 20k tokens
   plus mean KL — needs the end-to-end forward and runs at the Phase 5 gate. Per
   element the fp8 round trip costs ~2–3e-3 absolute on attention outputs of
   magnitude ~0.05, which is e4m3's 3-mantissa-bit resolution partly cancelling
   across the softmax average.
4. Paged KV is deferred to Phase 6, where the prefix cache actually needs it; a
   single sequence does not.

## The single riskiest open question

Now two, and they are related. The drafter question is unchanged. But Phase 4
adds: **G2 does not currently pass**, and the reason is structural rather than a
tuning miss — KV traffic at 64K is 2 GiB per token against 12.4 GiB of weights,
so attention is 23% of the step no matter how good the kernel is. The only
levers are kernel efficiency (49% → ~80% would take 64K attention from 5.26 to
3.2 ms and G2 to ~87%) or a smaller KV representation, and the second one trades
against the quality gate in §6.3.
