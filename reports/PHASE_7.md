# PHASE 7 — Speculative decoding, and the DFlash2 drafter

Status: **2.32x end-to-end decode over autoregressive**, greedy, with every
drafted token verified by the target. The drafter reproduces the official
z-lab/dflash reference's proposed path exactly.

| | measured |
|---|---|
| autoregressive decode, 4K ctx | 43.2 tok/s |
| DFlash2 speculative decode, mean of 3 prompts | **100.1 tok/s** |
| speedup | **2.32x** (best 2.82x, worst 1.69x) |
| mean accepted tokens per round | 4.10 / 5.79 / 6.83 (block 8) |
| verification block cost at T=8 | 2.11x one decode step |
| drafter size | 1.25 GiB at W4A16 (3.70 GiB bf16) |
| gate_dflash path vs reference | 7 / 7 |
| gate_spec, 4 prompts x 192 tokens | divergences are one bf16 ulp |

---

## 1. Three bugs that were live before any of this was fast

### The 20 GB allocation nobody checked

`gemv_scratch_alloc` sized its fp32 partial buffer as

```
max_splits * max_out * max_m  =  (82*16) * 248320 * 16 * 4 B  =  20 GB
```

`cudaMalloc` refused it. **The return value was never checked**, so `s.partial`
stayed null and every W4A16 GEMV in the model dereferenced it. The symptom was
`invalid argument` from a `cudaMemcpy` several hundred launches later, in
`model_generate_greedy`, which is nowhere near the cause.

The sizing was wrong on its own terms: `max_splits` and `max_out` never co-occur.
`gemv_choose_splits` stops as soon as `splits * ceil(out/ROWS)` reaches the fill
target, so `splits * out_f` is bounded by that target times ROWS_PER_BLOCK —
about 671k floats — and only the `splits == 1` case needs `max_out` at all. The
buffer is now ~59 MB, every malloc is checked, and every writer calls
`gemv_partial_check` before it launches.

### One int past the end

`argmax_scratch` was allocated at `256 * 8` bytes — exactly 512 ints — and
callers wrote the chosen token id at `+512`. `dalloc` is a bump allocator, so
that write landed on the first four bytes of whatever came next, which after the
Phase 7 edits was the attention workspace. It had "worked" for the whole project.

### Powers of two are not the reachable set

The small-M W4A16 and W8A16 GEMMs instantiated `M` in `{2,4,8,16}`. A block of
`k` drafts verifies `k+1` rows, so `k=7` aborted at runtime with
`gemm8: unsupported M 7`. All `M` in `[2,16]` are now instantiated.

The general lesson, again: `QWEN_DEBUG_SYNC=1` now synchronises after every stage
of `run_layer` and aborts at the first faulting kernel, and `=2` turns the same
hooks into a per-stage profile. Both were written to find these and both paid for
themselves immediately.

---

## 2. Making verification cheap enough to be worth doing

If verifying `T` tokens costs `T` decode steps there is nothing to gain. The
first measurement was not encouraging:

| T | ms | vs T=1 | expected speedup at p=0.78 |
|---|---|---|---|
| 1 | 22.22 | 1.00x | 1.00 |
| 2 | 64.21 | 2.89x | 0.62x |
| 8 | 66.14 | 2.98x | 1.32x |

A flat ~40 ms cliff the moment `T >= 2`, nearly independent of `T`. Three
candidates: CUDA-graph replay versus eager launches, the prefill-shaped attention
path, or the linear layers. Measured rather than guessed:

* **Eager T=1 is 22.81 ms against the graph's 21.63.** Launch overhead is 5%,
  not the cliff.
* **Forcing the decode attention kernel at T=8** (numerically wrong; a timing
  probe) took 65.95 -> 59.90 ms. Attention is ~6 ms of the 44.
* **Forcing cuBLAS instead of the tensor-core path** took it to 271 ms. cuBLAS
  dequantises the whole weight to bf16 first; it is 12.55x, not a fallback worth
  having at this shape.

So it was the tensor-core W4A16 GEMM, which was running at roughly half the
GEMV's bandwidth. The microbench says why:

| | ms at M=8 | effective GB/s |
|---|---|---|
| GEMV (M=1 reference) | 0.114 | 820 |
| MMA, as written | 0.227 | 411 |

**Stubbing out the MMA instructions changed nothing. Stubbing out the dequant
changed nothing. Stubbing out the weight loads changed almost nothing** — the
kernel still took 0.20 ms doing no global reads at all. That last one is the
interesting measurement: the kernel's *structure* cost more than the GEMV's
entire runtime, and no amount of tuning the inner arithmetic was going to help.

What did help, in the order it was found:

1. **k-splitting.** Each MMA block is 4 warps, so at `out_f=34816` the launch was
   544 blocks / 2176 warps against the GEMV's 8704. Splitting the k dimension
   across blocks with an fp32 partial and a reduce pass — exactly what the GEMV
   already did — brought 0.227 to 0.213.
2. **MMA_BN 64 -> 128**, eight warps per block: 0.213 -> 0.190. Half as many
   blocks re-staging the same activations.

Things that were tried and **lost**, recorded because the reasoning sounded good:

* Widening MMA_BK to 128 or 256 to get more loads in flight: 0.255 / 0.377. The
  shared-memory footprint costs more occupancy than the extra memory-level
  parallelism buys.
* A register prefetch of the next k-tile: no measurable change.
* Dropping the nibble reordering by permuting the *activations* into the
  magic-number's natural order instead (a real reduction in ALU work, and
  correct — the gate still passes): no measurable change. It is kept, because it
  is strictly less work for the same result, but it is not why the kernel got
  faster.

End result:

| T | ms | vs T=1 | expected speedup at p=0.78 |
|---|---|---|---|
| 1 | 21.73 | 1.00x | 1.00 |
| 2 | 41.64 | 1.92x | 0.93x |
| 4 | 44.42 | 2.04x | 1.40x |
| 8 | **45.81** | **2.11x** | **1.86x** |
| 16 | 48.53 | 2.23x | 2.00x |

---

## 3. The drafter

DFlash2 is not an ordinary draft model, and three of its properties are not
guessable from the shapes.

**It has no embedding and no lm_head.** The noise block is embedded with the
*target's* table and the logits come from the *target's* head. What the drafter
adds on top is a candidate selector: two rank-256 codebooks over the vocabulary
plus a rank-256 projection of its own hidden state, run as a first-order Markov
chain over the top-16 of the target's logits. That costs 16x256 MACs per position
instead of a second 248320-wide projection.

**Its keys and values come from the target's residual stream.** Layers
[5, 19, 33, 47, 61] are tapped, concatenated, and fused by `fc: 5*5120 -> 5120`.
Nothing the drafter itself predicted ever becomes a key. `model_enable_taps`
publishes those five layers straight into the layout `fc` consumes.

**Its attention is not causal.** `config.json` carries a top-level
`"is_causal": false`, and the reference then masks by
`|qpos - kpos| < sliding_window` in *both* directions. Reading
`layer_types == "sliding_attention"` as implying causality — which is what I did
first — gives an answer that looks entirely plausible and is 0.56 relative error
at layer 0. Bidirectional attention within the block is the whole point of block
diffusion: all eight noise positions see each other, so one forward emits the
entire block instead of eight autoregressive steps.

Smaller things the reference decides:

* `base_kernel` is `[2, kernel_size, hidden]` and the leading 2 is
  `{prepare, finish}`, **not** a third tap.
* `Qwen3RMSNorm` rounds the normalised value to bf16 *before* the weight
  multiply and has no `1.0 +`. The target's `Qwen3_5RMSNorm` has both. Using the
  wrong one is a 96%-out kind of mistake, and this project has already made it
  once, in Phase 3.

### How it is checked

`tools/dump_dflash_ref.py` drives `DFlash2DraftModel` itself, with forward hooks
on `fc`, `hidden_norm` and layer 0, so a mismatch names the stage rather than
just the output. Synthetic context hidden states are used deliberately: the
drafter's numerics do not depend on where the target's residual stream came
from, so the gate needs no 27B forward.

That staging is what found the causality bug in one run:

```
ctx_norm         rel 3.198e-03   OK
l0_ln            rel 0.000e+00   OK
l0_conv_prepare  rel 1.332e-03   OK
l0_attn          rel 5.630e-01   FAIL   <- here
```

**On the tensor tolerance.** Running the same reference in fp32 shows the
residual is the *reference's* bf16 rounding, not this implementation's error. At
every stage the CUDA output sits closer to fp32 than the bf16 reference does:

| | vs bf16 ref | vs fp32 ref |
|---|---|---|
| l0_attn | 9.45e-3 | 6.64e-3 |
| l0_out | 2.33e-2 | 1.66e-2 |
| draft_hidden | 2.80e-2 | 1.93e-2 |

So 6e-2 is the measured bf16 floor for five layers with two dynamic convolutions
each, not a number chosen to make the gate pass. The hard gate is the **path** —
the seven token ids the drafter actually proposes — and it matches 7/7. Candidate
set membership is reported but not gated: the two references disagree with each
other on 5 of 112 slots and on 2 of 7 path entries, because the 16th-ranked logit
is a near tie, and a drafter disagreement costs throughput, never correctness.

### Quantising it

W4A16 on the drafter's own matmuls was expected to trade acceptance for
bandwidth. It did not measurably trade anything:

| | bf16 | W4A16 |
|---|---|---|
| size | 3.70 GiB | **1.25 GiB** |
| mean speedup | 2.19x | **2.32x** |
| acceptance, p1 | 3.94 | 4.10 |
| acceptance, p0 / p2 | 5.79 / 6.83 | 5.79 / 6.83 |

W4A16 is now the default. Every drafted token is verified by the target, so
drafter quantisation error can only cost throughput.

---

## 4. Is it lossless?

Two separate claims, and they need to be kept apart.

**The acceptance rule is lossless by construction.** Longest-prefix argmax
equality plus the target's correction token emits exactly what the target would
have emitted, and `gate_spec` exercises the GDN state rollback that has to
accompany it.

**Batched verification does not sum in the same order as batch-1 decode.**
Verifying eight rows runs the tensor-core path, the prefill attention kernel and
the batched GDN scan; decoding one row runs the GEMV, the split-softmax decode
kernel and the single-step scan. All three pairs are equally valid and none of
them agree bit for bit. A near tie can therefore flip.

`gate_spec` measures this instead of asserting it. Over 4 prompts x 192 tokens,
one prompt diverges, under **both** drafters, and the gate teacher-forces the
common prefix to weigh the two candidates:

```
prompt 3, no-spec:  first divergence at 41
   279 (18.8750) vs 264 (18.7500)  gap 1.250e-01  bf16 ulp 1.250e-01 -> NEAR TIE
prompt 3, dflash:   first divergence at 28
  2530 (17.1250) vs 524 (17.0000)  gap 1.250e-01  bf16 ulp 1.250e-01 -> NEAR TIE
```

Both gaps are exactly one bf16 ulp: the two logits are adjacent representable
values. That both drafters land on the same prompt is the point — it is a
property of batched verification, not of DFlash2. The gate fails hard if a
divergence ever has a real logit gap, because that would mean the acceptance
rule or the rollback is broken.

The three other prompts are token-identical for 192 tokens under both drafters.

---

## 5. Never being slower

The suffix drafter lands 2.00 tokens per round on non-repetitive text. Against a
verification block that costs 1.92x a decode step at T=2, that is a net loss.
`spec_generate` now tracks realised tokens-per-round as an EWMA against the
measured cost curve (`1.876 + 0.0221*T`, fitted to the bench_block table above)
and falls back to plain decode when speculation stops paying, with a cooldown
that doubles on repeated failure and resets on success — so a prompt that turns
repetitive halfway through gets speculation back.

---

## 6. What is not done

* Speculation at 128K context is untested. The 128K peak is now 21.47 GB of 24
  (the scratch fix in section 1 took 1.7 GB off it), which leaves about 2.6 GB
  against a 1.25 GiB W4A16 drafter, so it plausibly fits — but "plausibly fits"
  is not a measurement and it is not claimed as one.
* The drafter's context push projects q as well as k and v, because it shares the
  fused qkv weight. On a long prefill that is wasted work.
* `k_attn_draft` is one warp per (head, query row), 32 blocks total. It is a
  small slice of a ~1.3 ms drafter and was written for clarity.
