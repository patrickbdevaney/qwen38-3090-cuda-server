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

### Revisited: the kernel was fixable, and a correction to how it is measured

The two deficits above were named from a stub experiment, so they were testable
rather than rhetorical. Both were real, and the cheap half of each has been
taken. But first, a correction, because the obvious way to score this kernel
flatters it badly.

**The "aggregate" line bench_gguf used to print is not the decode rate.** It
samples one tensor per quant type and pools them by bytes -- and `output.weight`
alone is 834 of the ~995 MiB sampled, so the number was very nearly a
measurement of Q5_K. Q5_K is 8.9% of this model. Weighting each type by the
bytes it actually occupies in the file gives a completely different picture:

| type | share of UD-Q3_K_XL bytes |
|---|---:|
| IQ4_XS | **37.9%** |
| IQ3_S | **26.9%** |
| Q5_K | 8.9% |
| IQ3_XXS | 6.9% |
| Q3_K | 6.6% |
| Q4_K | 4.9% |
| Q6_K | 2.6% |
| everything else | 5.3% |

Two thirds of this model is IQ4_XS and IQ3_S. bench_gguf now prints a
composition-weighted rate alongside the sampled one, and that is the number
quoted below.

Measured warm on an idle card, baseline = commit 460ff67's `gemv.cu` through
the identical bench:

| | baseline | now | |
|---|---:|---:|---:|
| Q5_K | 408.4 | 620.1 | +52% |
| Q3_K | 194.3 | 278.1 | +43% |
| IQ4_XS | 326.5 | 405.1 | +24% |
| IQ2_S | 229.6 | 267.8 | +17% |
| IQ3_S | 252.0 | 292.8 | +16% |
| IQ4_NL (untouched) | 363.6 | 363.2 | -- |
| sampled aggregate | 372.8 | 534.3 | +43% |
| **composition-weighted** | **285.8** | **354.8** | **+24%** |

Three changes, all bit-exact against ggml (gate_gguf_gemv, 13 types, zero
mismatching values):

1. **Q3_K scale extraction.** ggml's reference unpacks all sixteen six-bit
   scales out of a 12-byte field with a four-word shuffle. deq8 runs once per
   lane per super-block, so all 32 lanes were doing that whole unpack to read
   one byte of it. Deriving the single scale the lane needs is two byte loads
   and about six ALU ops.

2. **64-bit loads for Q4_K, Q5_K and IQ4_XS.** Each lane owns eight consecutive
   quantised bytes, fetched as eight 1-byte loads. Those three block formats are
   144, 176 and 136 bytes -- all multiples of 8 -- with their `qs`/`qh` fields at
   8-aligned offsets, and the lane offset is always a multiple of 8, so the eight
   bytes are one aligned 64-bit load with no repack. IQ4_XS is the important one:
   it is 38% of the file.

3. **`kmask_iq2xs` is powers of two.** Every i-quant path read `t.kmask[i]` from
   global memory eight times per deq8 to test a sign bit. Under `#pragma unroll`
   the index is a constant, so `1u << i` replaces a load with nothing.

The formats that did NOT get (2) are the ones whose block size is not a multiple
of 8 -- Q2_K (84), Q3_K (110), Q6_K (210) -- where consecutive blocks land on
rotating alignments and a 64-bit load would be illegal. Those need a padding
repack at load time (Q3_K 110 -> 112 costs 1.8% of the tensor), which is not done.

### Honest standing against llama.cpp

llama.cpp decodes UD-Q3_K_XL at 44.63 tok/s on this box. At 13.1 GB of weights
that is roughly **585 GB/s effective**, so at a composition-weighted 355 we are
at about **61% of llama.cpp**, up from 49%.

An earlier draft of this section claimed 88%, from the sampled aggregate. That
was wrong in our favour and is the reason the weighted metric now exists. The
remaining gap is dominated by the two i-quant types: IQ4_XS at 405 and IQ3_S at
293 against Q5_K's 620, and their grid/codebook lookups still go to global
memory rather than shared. That is the next lever, and it is untried.

### Codebook staging, and a measurement lesson that matters more

The i-quants are codebook quants: the stored index selects an entry from a fixed
grid in global memory, indexed once per value. Those grids are 1-8 KiB and every
warp in a block hits the same entries, so `k_gemv` now stages the grid its type
needs into shared memory once per block.

Measured warm, same session, same binary switched by a stash:

| type | share of file | global | shared | |
|---|---:|---:|---:|---|
| IQ3_S | 26.9% | 294.1 | **329.2** | +12% |
| IQ4_NL | 0.3% | 365.9 | 372.8 | +1.9% |
| IQ4_XS | 37.9% | 407.8 | 408.5 | -- |
| IQ2_S | 3.1% | 270.0 | 258.1 | **-4.4%** |
| **composition-weighted** | | **356.9** | **371.7** | **+4.1%** |

IQ4_XS gains nothing: its table is `kvalues_iq4nl`, sixteen bytes, which was
already resident in L1. IQ2_S *loses*: its grid is 8 KiB, four times the next
biggest, and the occupancy it takes costs more than the loads it saves. IQ2_S is
therefore not staged -- the measurement decided it, not the symmetry.

**The lesson that matters more than the +4%:** the first A/B run showed every
type slower, *including Q5_K and Q3_K, whose code did not change*. Two
consecutive runs of the identical binary measured 317.7 and 356.9 GB/s -- 12%
apart -- because the first run catches the card mid-clock-ramp. This box does
not permit `nvidia-smi --lock-gpu-clocks`, so **every GB/s figure in this repo is
only meaningful against another figure taken warm in the same session**, and a
cross-session comparison of two kernels can invent or hide a 12% effect. The
per-type numbers recorded earlier in this file were taken that way and hold up;
the discipline is now written down.

### The kernel was instruction bound, and saying so cost 42%

The staging experiment above answered a different question than the one that
mattered. Types with IDENTICAL memory access shapes were differing twofold --
Q5_K 624 GB/s, IQ3_S 329, Q3_K 246 -- and that cannot be bandwidth. Ranking the
types by how many MEMORY INSTRUCTIONS their per-lane dequantiser issues
reproduces the measured ranking exactly:

| type | loads per lane per run | GB/s (before) |
|---|---|---:|
| Q5_K | ~5 (two 64-bit + scales) | 624 |
| IQ4_XS | 4 + 8 table lookups | 426 |
| IQ3_S | ~6 + 2 lookups | 343 |
| Q3_K | ~19, all byte loads | 246 |
| Q6_K | ~17, plus a branch inside the unrolled loop | 163 |

Five changes, every one of them removing instructions rather than bytes, all
bit-exact against ggml:

1. **Activations were eight 2-byte loads** per lane per run. `base` is always a
   multiple of 8 and `in_f` of 256, so it is one 16-byte load -- unpacked with
   shifts rather than through a local array, which would spill the vector and
   undo the point.
2. **IQ4_XS and IQ4_NL keep their table in registers.** `kvalues_iq4nl` is
   sixteen signed bytes; two `PRMT` and a `__vcmpgeu4` blend replace four
   lookups with no memory traffic at all. This is also why staging it in shared
   bought nothing: sixteen bytes read by thirty-two lanes at thirty-two offsets
   conflicts on nearly every bank, so an LDS costs what the L1 hit cost.
3. **IQ3_S and IQ3_XXS** take their two grid indices from one 16-bit load, and
   the sign folds into the INTEGER before conversion -- negating an int and
   negating the float it converts to agree exactly, so it is bit-exact and drops
   a multiply per element. The same fold covers IQ2_XS / IQ2_XXS / IQ2_S.
4. **Q2_K, Q3_K and Q6_K get 16-bit loads.** They were excluded from the earlier
   64-bit work because 84, 110 and 210 are not multiples of 8 -- but they are all
   EVEN, and every field offset in these paths is a multiple of 8, so 16-bit is
   always legal. Sixteen byte loads become eight. The padding repack the earlier
   note assumed was necessary is not.
5. **Q6_K decided its sub-block offset, nibble shift and scale inside the
   unrolled loop**, none of which depend on the element index.

| type | share | before | after | |
|---|---:|---:|---:|---:|
| Q6_K | 2.6% | 163 | **771** | +373% |
| IQ2_XXS | 0.4% | 186 | 313 | +68% |
| IQ3_XXS | 6.9% | 286 | 462 | +62% |
| Q2_K | 0.5% | 273 | 453 | +66% |
| IQ2_XS | 0.8% | 236 | 376 | +59% |
| IQ3_S | 26.9% | 343 | 508 | +48% |
| IQ4_NL | 0.3% | 362 | 533 | +47% |
| IQ2_S | 3.1% | 267 | 383 | +43% |
| Q8_0 | 0.3% | 488 | 684 | +40% |
| Q4_K | 4.9% | 465 | 580 | +25% |
| IQ4_XS | 37.9% | 426 | 524 | +23% |
| Q5_K | 8.9% | 618 | 658 | +6% |
| **composition-weighted** | | **354.8** | **503.8** | **+42%** |

End to end: 23.8 -> 30.6 tok/s of autoregressive decode at 4096 context, 42% of
the DRAM roofline against 33%.

`bench_gguf` had been measuring a hardcoded list of six tensors that left seven
of the fourteen types unsampled, so its composition-weighted number covered 84%
of the model and silently assumed the rest behaved like the average of what was
measured. It now picks the largest tensor of every type present -- which is how
Q6_K's 163 GB/s came to light at all, at 2.6% of the bytes and 13% of the time.

### What is still on the table, and why it is worth taking

At 504 GB/s the GEMV is at 55% of measured DRAM, against the AWQ path's 84%. The
remaining gap is structural rather than another peephole: **AWQ is repacked at
load so that 32 lanes read 128 contiguous bytes with each lane owning a different
ROW**, while the GGUF kernel gives one warp one row and splits a 110-byte block
across its 32 lanes. That means the block header -- scales, mins, the super-block
d -- is decoded once per eight elements per lane instead of once per block, a
32-fold redundancy no amount of load-width tuning removes.

Interleaving 32 rows at load, so a lane owns a whole block and the warp's reads
are coalesced across rows, would cut memory instructions per element by roughly
an order of magnitude and amortise the header the same way AWQ does. It costs a
repack pass at load, a padding of block sizes to a multiple of 4 (under 2%), and
a rewrite of the kernel's addressing. The prize is quantified: **GGUF's roofline
is 72.6 tok/s against AWQ's 63.4**, so a kernel at AWQ's efficiency makes the
smaller model the faster one.

This does not change the *headroom* conclusion above -- INT4 KV still frees 3.7x
more than Q3_K_XL would. It does mean the GGUF path is no longer embarrassing,
but "AWQ and GGUF are viable substitutes" is not yet true and should not be
claimed: the kernel is at 61% of the reference implementation, and `model.cu`
still has no GGUF loader, so nothing runs end to end.

## CORRECTION: the loader exists, and it changes the verdict above

Everything from "The Q3_K_XL fused GEMV: built, correct, and NOT competitive"
down was written with no way to run a GGUF end to end, so it could only compare
*kernel efficiency* and had to assume the accuracy question away. Both halves of
that turn out to be wrong in ways that matter.

**The prediction in this file was that AWQ and a Q4-class GGUF would be within
noise, and that Q3_K_XL was "a genuinely different point on the curve".** The
first half is right and the second is not. Measured through this server against
the original BF16 weights, over 15520 teacher-forced positions:

| | KL mean | KL p99 | top-1 | weights |
|---|---:|---:|---:|---:|
| AWQ INT4 g128 + INT8 head | 1.495e-01 | 4.263 | 98.56% | 13.06 GiB |
| **GGUF UD-Q3_K_XL** | **1.432e-01** | 4.274 | **98.57%** | **11.36 GiB** |

Q3_K_XL is not a different point on the accuracy curve. It is the *same* point,
1.70 GiB lighter. The reason to expect otherwise -- 3 bits against 4 -- does not
survive contact with a dynamic quant that spends its bits per tensor: two thirds
of this file is IQ4_XS and IQ3_S, and the head is Q5_K.

**"That prize is no longer worth it, because INT4 KV already freed 3.58 GiB"**
was also the wrong frame. The two are not alternatives, they compose. INT4 KV
and Q3_K_XL together free 3.58 + 1.70 GiB, and the 1.70 is worth 99K tokens of
context on top of what the KV format already bought. The argument that only one
of them was needed came from treating headroom as a threshold to clear rather
than a budget to spend.

What survives unchanged is the *speed* half: the GGUF decode kernel is still at
about 61% of llama.cpp's effective bandwidth, so the smaller model decodes
slower than AWQ despite reading fewer bytes. That is the trade the two formats
now present -- see `reports/BENCHMARKS.md` for the measured curve -- and it is a
real choice for the user rather than a defect in one of them: AWQ for
throughput, GGUF for context. Closing the 61% is what would make the choice
one-sided, and it is still the highest-value open kernel item in the project.
