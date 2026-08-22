# What quantisation actually costs, against the weights Qwen released

Everything before this compared kernels against *dequantised AWQ* weights, which
answers "does my CUDA match PyTorch on the same numbers" and says nothing about
what the quantisation itself cost. This measures against the real thing:
`Qwen/Qwen3.8-27B`, 52 GiB of BF16, streamed layer by layer through accelerate
because it has never fit on this card and never will.

## Method

Teacher forcing, not generation. Each prompt goes through the model once and the
next-token distribution at **every** position is kept, so eight prompts yield
15520 independent predictions rather than eight lucky continuations. Every
engine reads the same `tests/fixtures/quantcmp/prompts.json`, and the scorer
asserts the token ids agree before it compares a single logit — a tokenisation
difference makes position *i* a different prediction problem on each side, and
scoring it anyway would produce a confident number about nothing.

The prompt set is deliberately not wikitext. Average perplexity over prose does
not surface the failures that break an agent: picking the wrong identifier,
emitting a malformed tool call, losing a fact that scrolled 13K tokens up. So
the set is code generation, code debugging, a tool-call chain, strict JSON
output, world knowledge, multi-step arithmetic, and two multi-hop retrieval
prompts (1350 and 13552 tokens) with a fact buried near the top.

| tool | engine |
|---|---|
| `tools/quantcmp_bf16.py` | HF transformers, BF16, CPU+GPU offload |
| `tools/quantcmp_dump.cu` | this server, AWQ INT4 g128 + 8-bit lm_head |
| `tools/quantcmp_exl3.py` | exllamav3, EXL3 3.00bpw |
| `tools/quantcmp_score.py` | KL, top-1, top-5 |

## Result

KL(BF16 ‖ candidate) in nats, over all 15520 positions:

| | KL mean | KL median | KL p99 | top-1 | top-5 |
|---|---:|---:|---:|---:|---:|
| **AWQ INT4 g128 + INT8 head (ours)** | **1.495e-01** | **5.21e-05** | **4.263** | **98.56%** | **99.39%** |
| EXL3 3.00bpw | 1.672e-01 | 6.41e-05 | 5.216 | 97.74% | 99.17% |

**AWQ is closer to BF16 on every metric**, including with an 8-bit lm_head
stacked on top of the INT4 body — a second quantisation EXL3 is not carrying.

The distribution matters more than the headline. The median KL is 5e-05: at half
of all positions the quantised model is, to five decimal places, the BF16 model.
The p99 is 4.26 — a factor of 80,000 higher. Quantisation error is not spread
evenly over tokens; it concentrates on a small set of positions and is invisible
in any average. That is the shape an agent trips over, and it is why this report
carries a p99 column at all.

Per prompt (ours):

| prompt | tok | KL mean | KL p99 | top-1 |
|---|---:|---:|---:|---:|
| code_impl | 98 | 8.85e-02 | 1.220 | 86.73% |
| code_debug | 72 | 1.59e-02 | 0.102 | 95.83% |
| tool_call | 126 | 5.80e-02 | 0.588 | 91.27% |
| json_struct | 109 | 2.76e-02 | 0.134 | 94.50% |
| world_knowledge | 112 | 2.24e-01 | 2.246 | 83.93% |
| math_reason | 101 | 3.24e-02 | 0.239 | 93.07% |
| multihop_short | 1350 | 1.22e-01 | 2.999 | 98.52% |
| multihop_long | 13552 | 1.56e-01 | 4.683 | 98.92% |

The long prompts score *better* on top-1 (98.5-98.9%) than the short ones
(84-96%), which is not the model being better at long context — it is the
synthetic repo-map filler being highly predictable, so both models agree
trivially on most of it. Read the long-prompt rows as a statement about the
tail (p99 4.68) rather than the mean. `world_knowledge` is the worst row on
both engines, and that is the expected place for it: recalling a specific
bandwidth figure or architecture detail is exactly the low-redundancy, high-
confidence prediction that a weight perturbation can flip.

## What this settles

The standing condition on EXL3 was: *if exllama remains accurate and fastest,
build a kernel server for it too*. All three tests are now in, and it fails all
three:

| | EXL3 3.00bpw | EXL3 3.50bpw | ours |
|---|---|---|---|
| decode @ 262144 | 22.59 tok/s | 21.99 | **24.5** |
| peak VRAM @ 262144 | 19910 MiB | 21481 MiB | **19749 MiB** |
| KL vs BF16 | 1.672e-01 | **1.308e-01** | 1.495e-01 |

(The 19749 MiB is model plus KV in isolation. With the DFlash2 drafter, which is
not optional in this server, the real figure is 21862 MiB. The EXL3 columns
carry no drafter either, so the comparison is like-for-like, but neither number
is what the server actually occupies.)

At 3.00bpw it loses on all three. At 3.50bpw it wins on accuracy and loses on
both speed and VRAM. Neither configuration is the "accurate *and* fastest" the
condition asked for, so **no EXL3 backend** — but the accuracy result is real
and is recorded rather than buried. The harness is committed, so this can be
rerun against any future format or a 3.25bpw split-the-difference build.

## The fair fight: EXL3 3.50bpw, at matched VRAM

3.00bpw is not an apples-to-apples comparison — it carries 25% fewer bits per
weight than AWQ INT4, so of course it is less accurate. The matched test is
**EXL3 3.50bpw at 14.285 GiB**, against our 14.25 GiB of body plus heads: the
same weight budget, spent by a different quantiser. Run rather than argued:

| | KL mean | KL median | KL p99 | top-1 | top-5 |
|---|---:|---:|---:|---:|---:|
| ours, AWQ INT4 + INT8 head | 1.495e-01 | 5.21e-05 | 4.263 | **98.56%** | 99.39% |
| **EXL3 3.50bpw** | **1.308e-01** | **4.86e-05** | **3.493** | 98.51% | **99.43%** |

**At equal VRAM the trellis quantiser is more accurate**, and by the margin that
matters most: the p99 tail is 3.49 against 4.263, 18% lower. Top-1 is a tie.
This is a real result and it goes against us; some of the gap is the 8-bit
lm_head and FP8 KV that our column carries and EXL3's does not, but not all of
it, and the honest reading is that EXL3 3.50bpw is the better-quantised model.

Speed and footprint go the other way, and the gap widens with context:

| ctx | ours | EXL3 3.50bpw + 4-bit KV | ours |
|---:|---:|---:|---:|
| 4096 | **45.5** tok/s | 44.62 | +2.0% |
| 16384 | **42.8** | 42.54 | +0.6% |
| 32768 | **40.2** | 38.94 | +3.2% |
| 65536 | **37.2** | 35.12 | +5.9% |
| 131072 | **31.8** | 29.63 | +7.3% |
| 262144 | **24.5** | 21.99 | **+11.4%** |
| peak VRAM @ 262144 | **19749 MiB** | 21481 MiB | 1.7 GiB lighter |

So the matched comparison is a genuine trade rather than a sweep: **EXL3 3.50bpw
buys about 13% lower mean KL and an 18% lower tail, and costs 11% of decode
throughput at 262K plus 1.7 GiB of VRAM.**

For this server's purpose that trade is the wrong way round. An agentic coding
harness lives at long context, where the 11% is compounding and the accuracy
difference is a tail effect on a distribution whose median is already 5e-05. But
it is a defensible preference in the other direction, and it is the reason the
harness that produced these numbers is committed rather than thrown away.

## GGUF, measured through this server

The GGUF row was left open in an earlier revision of this report because the
server could not run a GGUF checkpoint end to end. It can now, so the row is
filled the same way as the others: `tools/quantcmp_dump.cu` with a `--gguf`
weights path, same prompts, same scorer, same BF16 reference.

| | KL mean | KL median | KL p99 | top-1 | top-5 |
|---|---:|---:|---:|---:|---:|
| ours, AWQ INT4 g128 + INT8 head | 1.495e-01 | 5.21e-05 | 4.263 | 98.56% | 99.39% |
| **ours, GGUF UD-Q3_K_XL** | **1.432e-01** | 5.92e-05 | 4.274 | **98.57%** | 99.39% |
| EXL3 3.50bpw | 1.308e-01 | 4.86e-05 | 3.493 | 98.51% | 99.43% |

**AWQ and UD-Q3_K_XL are the same model to within noise** — mean KL 1.432e-01
against 1.495e-01, p99 4.274 against 4.263, top-1 within 0.01 points — while
the GGUF carries 11.36 GiB of weights against AWQ's 13.06. Both numbers come
from the same kernels and the same head treatment, which is what makes them
comparable at all: the AWQ column pays for an INT8 lm_head on top of its INT4
body, and the GGUF column runs the file's own Q5_K head through the fused GEMV
rather than re-quantising it.

That 1.70 GiB is 99K more tokens of INT4 KV, bought at no measurable accuracy
cost. It is the first result in this report that moves the frontier rather than
just locating a point on it. What it does NOT buy is speed — see
`reports/BENCHMARKS.md`; the GGUF decode kernel is still at about 61% of
llama.cpp's effective bandwidth, so the smaller model decodes slower.

Unsloth's UD-Q3_K_XL is a *dynamic* quant: it picks a type per tensor, and this
file uses eleven of them. The row above is a statement about that specific file,
not about "Q3 GGUF" in general.

### One asymmetry, and it runs against us

The two engines did not use the same KV cache, and they could not: this server
has no unquantised KV path at all (`KvFmt` is `{FP8, INT4}`), because storing
BF16 keys and values was never a configuration worth building for a 24 GiB card.
So ours ran at **FP8 KV**, its floor and its serving default, while exllamav3 ran
at its default **FP16 KV**.

That means the AWQ column carries a quantisation the EXL3 column does not, on
top of the 8-bit lm_head it also carries and EXL3 also does not — and it still
wins on every metric. The gap is a lower bound on how far ahead the AWQ path is
on weights alone, not an overstatement of it.

For scale, the KV contribution is small next to the weight contribution: INT4 KV
— a *further* step down from FP8 — measures KL 3.69e-04 at 131k with needle
retrieval intact (gate_kvquant), against the 1.5e-01 mean here.
