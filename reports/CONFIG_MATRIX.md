# Choosing the config: what to measure, and what each measurement can honestly say

The end state is one recommended configuration. Getting there means separating
questions that need different oracles and have different confidence levels.

## The two axes

**Weights**: AWQ INT4 g128 (what we run) vs a GGUF quant.
**KV cache**: FP8 e4m3 (what we run) vs INT4 with per-32-group scales.

They are independent -- both weight formats get the same KV benefit -- so they
can and should be measured separately.

## Axis 1: KV precision. Cleanly measurable, and already in flight.

This is a true A/B: identical weights, identical kernels, identical prompt, one
variable. No external oracle needed, because FP8 **is** the reference.

| format | bits/elem | per token | at 262144 |
|---|---|---|---|
| FP16 | 16 | 64 KiB | 16.0 GiB -- does not fit, at all |
| FP8 e4m3 (today) | 8 | 32 KiB | 8.0 GiB |
| INT8 + group scale | 8.5 | 34 KiB | 8.5 GiB |
| **INT4 + group scale** | **4.5** | **18 KiB** | **4.5 GiB** |

FP16 is off the table on 24 GB and FP32 is not worth discussing, so the ladder
really is FP8 vs INT4 -- your read was right. One nuance worth flagging: **INT8
with a group scale may be strictly more accurate than FP8 e4m3 at the same
size**, because e4m3 spends 4 bits on exponent and keeps only 3 mantissa bits,
while INT8 against a per-32 absmax keeps 7 bits of mantissa over a narrower
range. That is a possible free accuracy win rather than a size win, and it is
cheap to test with the machinery now in place.

What is measured, weakest to strongest:

1. top-1 agreement and KL against the FP8 run,
2. how far the greedy token stream stays identical,
3. **needle retrieval** -- a fact buried at the start of a long context, asked
   about at the end.

(3) is the one that decides it. Averaged token statistics *understate* KV damage
badly, because the harm is concentrated on the few tokens where the model has to
look something up; everywhere else it can talk fluently around a key it did not
resolve. `gate_kvquant` therefore treats retrieval as pass/fail, and refuses to
report a failure if the FP8 baseline itself misses the needle -- that would be an
inconclusive test, not a quantisation result.

## Axis 2: AWQ vs GGUF weights. Needs an oracle we do not have yet.

This one cannot be done honestly with what is on the box. The trap:

> Our measured 6.99e-4 KL is against `Qwen3.8-27B-BF16-from-INT4`, which is
> **dequantised from the AWQ checkpoint**. It measures KERNEL FIDELITY, not
> AWQ's quantisation loss. Comparing a GGUF against it would flatter AWQ by
> construction, because AWQ's own error is baked into the "reference".

A real answer needs true BF16, which we do not have locally -- only the AWQ
checkpoint and a dequantisation of it. Two routes:

**(a) llama.cpp's own KL tooling.** Convert or fetch an F16 GGUF (~54 GB; 476 GB
free), dump reference logits once, then score each candidate:

```bash
llama-perplexity -m Qwen3.8-27B-F16.gguf   -f corpus.txt --kl-divergence-base base.dat
llama-perplexity -m Qwen3.8-27B-UD-Q3_K_XL.gguf -f corpus.txt --kl-divergence-base base.dat --kl-divergence
```

Gives per-quant KL and top-1 agreement on THIS model. It cannot score our AWQ
path, because that is not a GGUF -- so it ranks GGUF quants against each other
and against F16, and AWQ has to be placed separately.

**(b) Fetch the original BF16 safetensors (~54 GB) and run OUR forward against
it**, which is what `gate_forward` already does -- just against a real reference
instead of a self-derived one. That is the only route that puts AWQ and GGUF on
the same axis, measured through the same kernels.

(b) is the honest one. It is also the one that finally answers "what did AWQ
actually cost", which nobody in this project has measured.

## What the answer is likely to be, stated as a prediction

Recording this in advance so it can be checked rather than rationalised:

* KV INT4 will pass needle retrieval at 8K and get *worse* as context grows,
  because the error is in the keys and the number of competing keys grows.
  If it fails anywhere it will fail at 128K+ first.
  **MEASURED: correct on the trend (KL 9.2e-05 at 8K -> 3.69e-04 at 131k, 4x),
  wrong on the consequence -- retrieval survived at 131k and the KL is still
  below what the INT4 weights already cost.**
* `K FP8 / V INT4` will be nearly free -- values are averaged over the attention
  distribution, so their noise cancels; keys are not.
  **MEASURED: correct. V-only is ~2x cleaner than both sides at every length.**
* AWQ INT4 g128 and a Q4-class GGUF will be within noise of each other, because
  they are the same bit budget. **UD-IQ4_XS is already measured to be the same
  size as our body (11.941 vs 11.859 GiB), so there is nothing to win there.**
  The interesting GGUF comparison is Q3_K_XL, which is a genuinely different
  point on the curve.

## The decode-speed axis

Decode is bandwidth bound, so speed follows bytes read per token almost exactly
-- our AWQ measurement (13.06 GiB/token -> 45.8 tok/s) is the calibration.

**PREDICTED WRONG.** I expected INT4 KV to make 262144 decode faster as well as
smaller, since it halves KV traffic. Measured: 24.4 -> 24.5 tok/s, i.e.
unchanged. The decode attention kernel was already running at 444 GB/s against a
load-only ceiling of 788 GB/s, so it is latency and ALU bound rather than
bandwidth bound, and the extra dequantisation exactly consumes the bandwidth
that was freed. **KV quantisation buys memory, not speed** -- which also means
the attention kernel has ~340 GB/s of headroom still on the table for whoever
fixes it.

| config | weights+head | KV @ 262144 |
|---|---|---|
| AWQ + FP8 KV | 13.06 GiB | 8.0 GiB |
| AWQ + INT4 KV | 13.06 GiB | 4.5 GiB |
| Q3_K_XL + INT4 KV | 11.72 GiB | 4.5 GiB |

## The Q3_K_XL fused GEMV: built, correct, and NOT competitive

The fused GEMV over GGUF blocks is done and gated -- all 14 block types produce
values bit-identical to the ggml-verified dequantiser, and the dot products land
at the bf16 output floor (~1.6e-03 relative). It reads the quantised blocks
directly, with no dequantised round trip.

It is also **2.2x less efficient per byte than the AWQ path**, which more than
cancels Q3_K_XL's 8% smaller body:

| | GB/s | of 914.2 |
|---|---|---|
| AWQ INT4 g128 (ours today) | **769.8** | 84.2% |
| GGUF fused, dequant stubbed out | 646.0 | 70.7% |
| GGUF fused, as built | **347.3** | 38.0% |

Sweeping warps per block from 2 to 16 moved nothing (353 / 353 / 340 / 353), so
it is structural rather than occupancy. The stub isolates two separate deficits:

1. **Layout, ~16%.** GGUF stores blocks row-major, so a warp streaming one row
   reads 110-176 contiguous bytes at a time. AWQ is repacked so 32 lanes read
   128 contiguous bytes with each lane owning a different ROW -- one transaction
   instead of a small one.
2. **Dequant ALU, the other ~46%.** Each lane owns 8 of a 256-element run, so
   the block header and sub-block scale work is repeated up to 32x per block.
   The per-type spread shows it directly: Q3_K, whose deq8 unpacks 12 scale
   bytes into 16 six-bit scales on every lane, is worst at 172 GB/s; Q5_K, whose
   scale extraction is two shifts, is best at 383.

What it would cost to fix: a per-type repack into a lane-friendly layout (1),
plus cooperative per-warp scale computation broadcast by shuffle (2). Best case
is parity with AWQ's 84%, which buys 8% faster decode and 0.95 GiB.

**That prize is no longer worth it, because INT4 KV already freed 3.58 GiB** --
3.7x more than Q3_K_XL would, at a measured KL of 3.69e-04 that leaves needle
retrieval intact at 131k tokens. The headroom problem is solved on the other
axis.

So: the GGUF stack stays in the tree, complete and verified, as the foundation
for a future quant that actually pays. Q3_K_XL is not that quant at this kernel
efficiency, and that is a measurement rather than an opinion.

### Revisited: the kernel was fixable, and 40% of the gap is now closed

The two deficits above were named from a stub experiment, so they were testable
rather than rhetorical. Both turned out to be real, and the cheap half of each
has now been taken:

| | GB/s | of 914.2 |
|---|---|---|
| GGUF fused, as first built | 347.3 | 38.0% |
| **GGUF fused, today (cold, first run)** | **482.6** | **52.8%** |
| GGUF fused, today (warm, 5th run) | 524.1 | 57.3% |
| GGUF fused, dequant stubbed out | 646.0 | 70.7% |
| AWQ INT4 g128 | 769.8 | 84.2% |

Two changes, both bit-exact against ggml (gate_gguf_gemv, all 13 types, zero
mismatching values):

The two rows for "today" are the same binary: five consecutive runs measure
482.6 / 493.5 / 516.6 / 517.0 / 524.1 GB/s as the card settles at sustained
clocks. The 347.3 baseline was a single cold run, so **347 -> 483 is the
like-for-like comparison**, +39%. Per-type numbers below are cold-run.

1. **Q3_K scale extraction: 172 -> 246 GB/s.** ggml's reference unpacks all
   sixteen six-bit scales out of a 12-byte field with a four-word shuffle. This
   kernel calls deq8 once per lane per super-block, so all 32 lanes were doing
   that whole unpack to read one byte of the result. Deriving the single scale
   the lane needs is two byte loads and about six ALU ops.

2. **Q4_K and Q5_K 64-bit loads: Q5_K 392 -> 570 GB/s cold, 620 warm.** Each lane owns eight
   consecutive quantised bytes, which were being fetched as eight 1-byte loads.
   Q4_K and Q5_K blocks are 144 and 176 bytes -- both multiples of 16 -- with
   their `qs`/`qh` fields at 8-aligned offsets, and the lane offset is always a
   multiple of 8, so those eight bytes are one aligned 64-bit load with no
   repack at all. This is the single biggest lever in the file, because
   `output.weight` is 834 MiB of Q5_K, more than every other tensor in the
   sweep combined.

The formats that did NOT get this are the ones whose block size is not a
multiple of 8 -- Q2_K (84 bytes), Q3_K (110), Q6_K (210) -- where consecutive
blocks land on rotating alignments and a 64-bit load would be illegal. Those
need a padding repack at load time (Q3_K 110 -> 112 costs 1.8% of the tensor),
which is the obvious next step and is not done.

For scale: llama.cpp runs Q3_K_XL at an effective 546 GB/s. At 483 cold we are
at **88% of llama.cpp**, up from 64%; warm, the aggregate crosses it. Comparing
a kernel microbenchmark against an end-to-end effective rate is apples to
oranges in llama.cpp's favour, so treat 88% as the floor and do not claim
parity until the GGUF path runs end to end -- which it does not yet, because
model.cu has no GGUF loader. The remaining structural gap to AWQ is the
row-major block layout (deficit 1), which no amount of per-type arithmetic will
fix.

This does not change the *headroom* conclusion above -- INT4 KV still frees 3.7x
more than Q3_K_XL would -- but it does change the substitutability one. A GGUF
path within 11% of llama.cpp is a credible weight option to offer users rather
than a curiosity, which is what it was at 347.

### Is 347 GB/s the FORMAT or MY KERNEL? Measured: mostly my kernel, and it
### does not change the conclusion.

Two checks rather than an assumption.

**Speculation amortises the dequantisation, as predicted.** At M rows the weight
stream is read once and the per-block header work is shared, so an ALU-bound
kernel should improve with M. On blk.1.ffn_up (Q3_K):

| M | ms | per-token ms |
|---|---|---|
| 1 | 0.208 | 0.2079 |
| 8 | 0.724 | **0.0905** |

2.3x better per token. So yes -- with DFlash2 running blocks of 8, the gap
narrows substantially.

**llama.cpp's mature K-quant kernels on this box, same file:**

| | size | tok/s | effective GB/s |
|---|---|---|---|
| llama.cpp Q4_K_M | 17.67 GiB | 38.41 | 679 |
| llama.cpp **UD-Q3_K_XL** | 12.23 GiB | **44.63** | 546 |
| **ours, AWQ INT4 g128** | 13.06 GiB/token | **45.8** | 598 |

So a *mature* Q3_K_XL implementation reaches 44.63 tok/s -- still slightly below
our AWQ at 45.8, despite reading 1.34 GiB less per token. The 3-bit dequant
costs llama.cpp 20% of its bandwidth efficiency (679 -> 546), which eats the
size advantage there too.

My kernel at 347 GB/s is therefore genuinely immature relative to llama.cpp's
~546, and closing that is a real (if unrewarding) engineering task. But the
conclusion is unchanged and now rests on someone else's mature kernels rather
than mine: **Q3_K_XL does not buy decode speed on this card.** It buys 0.95 GiB,
and INT4 KV already bought 3.58.

### Attempt at llama.cpp parity, and why it failed

The obvious fix for the 46% dequantisation overhead is to give each lane a whole
32-element sub-block instead of 8 elements, so the block header (and for Q3_K a
12-byte unpack into sixteen 6-bit scales) is computed once per 32 rather than
once per 8. Implemented, and it stayed bit-exact on all 13 types.

**It made things 2.6x WORSE: 347 -> 132 GB/s.**

Not register spilling -- ptxas reports 0 bytes spilled. The cause is the
ACTIVATION reads. In the 8-element layout, one load instruction has the 32 lanes
touching addresses 16 bytes apart, spanning 512 bytes -- 8 cache lines. With 32
consecutive elements per lane they are 64 bytes apart, spanning 2048 bytes -- 32
cache lines, one transaction per lane. Improving the weight access made the `x`
access four times worse, and `x` is read for every element while a weight byte is
read once.

The remaining route is per-type vectorised byte loads: our AWQ kernel gets its
speed partly by loading 4 bytes at a time and shifting out nibbles, instead of
eight single-byte loads. That is blocked here by alignment -- a Q5_K block is 176
bytes but its `qs` array starts at byte 46, so a lane's 8 bytes are not 4-byte
aligned. Fixing it needs a format-preserving repack at load time that pads
blocks and reorders fields to align the quant arrays, at a few percent size cost.

So llama.cpp parity is reachable but it is a per-type engineering project --
which is, fairly, what llama.cpp's K-quant kernels ARE. Recorded rather than
hand-waved: the current kernel is 347 GB/s, the mechanism of the gap is
understood and measured, and the next step is specified.

## The Ampere constraint that applies to every format

sm_86 has **no hardware for sub-8-bit types**. Every 4-bit format pays a
software dequantisation, and the binding question is always ops-per-byte, not
bits-per-weight:

* our AWQ path is ~2.5 ops/weight using the bf16 magic-number trick -- about as
  cheap as this arithmetic gets, which is why it holds 84.2% of DRAM,
* K-quants cost more (six-bit sub-block scales, mins, high-bit merges) and
  llama.cpp lands at 546-679 GB/s,
* going *lower* in bits raises ops-per-byte, because the same header work is
  spread over fewer bytes.

That is the mechanism behind every measurement in this file, and it is why
"smaller quant" has not once translated into "faster decode" on this card.

## Why this matters beyond the numbers

The point of the headroom is **RoPE-extended context**. 262144 is the trained
maximum; going past it needs YaRN-style extrapolation, and extrapolation quality
degrades with distance -- so it is only worth attempting from a config that has
memory to spare AND has been shown not to lose retrieval at the trained length.
A config that already misses needles at 262144 will not survive being stretched
to 512K.
