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

| | EXL3 3.00bpw | ours |
|---|---|---|
| decode @ 262144 | 22.59 tok/s | **24.5** |
| peak VRAM @ 262144 | 19910 MiB | **19749 MiB** |
| KL vs BF16 | 1.672e-01 | **1.495e-01** |

So no EXL3 backend. That is a decision made on measurements, and the harness
that produced them is committed, so it can be rerun against any future format.

**In fairness to the format**: EXL3 3.00bpw carries 25% fewer bits per weight
than AWQ INT4 and lands within 12% on mean KL. Per bit, the trellis quantiser is
clearly the better one. It loses here on deployment terms, not on quantiser
quality — at the VRAM this box has, the extra bits are free and the format that
uses them wins. The apples-to-apples test is EXL3 **3.50bpw**, whose 14.29 GiB
matches our 14.25 GiB of body plus heads almost exactly; that comparison is
queued and not yet run.

## Not covered

GGUF has no number here, because this server still cannot run a GGUF checkpoint
end to end — `model.cu` has no GGUF loader, only the container parser, the
dequantiser and the fused GEMV, all gated in isolation. Producing a GGUF row
means either writing that loader or comparing llama.cpp against a BF16 GGUF,
which measures llama.cpp rather than us. Left open rather than faked.

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
