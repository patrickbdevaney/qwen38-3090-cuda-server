# What this server is for, and what is left

## North star

A lean CUDA/C++ inference backend that is a genuine production substitute for
vLLM, llama.cpp or SGLang **for one specific job**: powering a local agentic
coding harness — a local Claude Code / Codex / OpenCode — on one RTX 3090.

Not a general serving stack. One model, one card, one sequence, specialised hard,
and good enough in practice that the local option is close to the proprietary
one for real engineering work.

That job has a particular shape, and every primitive in this repo exists because
of it:

| the harness does this | so the server needs |
|---|---|
| long multi-turn sessions over a repo | **262K context**, and a KV cache small enough to hold it |
| re-sends the whole conversation every turn | **prefix caching** — turn N+1 must not re-read turn N |
| long tool-call chains, one wrong argument kills the run | **quantisation that does not degrade tool/code tokens** |
| reads files, repomaps, RAG chunks, memory .md files | **fast prefill**, because context is mostly re-read |
| compacts context at ~75% and writes memory out | **cheap state snapshots**, so compaction is not a full re-prefill |
| waits on every token | **decode speed** — the thing the user actually feels |

The hybrid architecture is what makes this reachable on 24 GB at all: only 16 of
64 layers hold KV, so a token costs 32 KiB at FP8 instead of 128 KiB, and the
GatedDeltaNet state is constant in context length. A dense 27B would need 4x the
KV and 262K would not be on the table.

## Where it stands

| | |
|---|---|
| autoregressive decode, 4K | 45.8 tok/s (llama.cpp 38.41) |
| DFlash2 speculative decode | 100.1 tok/s mean, 133.3 through the server |
| context | **262144**, the model's trained maximum |
| prefix cache, 2nd turn | 90.5x faster prefill |
| gates | 16 / 16 |

## The KV cache is the bigger lever, and it was hiding in plain sight

At 262144 context the KV cache costs **8.0 GiB** -- more than half the memory
pressure -- and it was at 8 bits while the weights were at 4.

| KV format | per token | at 262144 | frees |
|---|---|---|---|
| FP8 e4m3 | 32 KiB | 8.0 GiB | -- |
| K FP8 / V INT4 | 25 KiB | 6.3 GiB | 1.7 GiB |
| INT4 / INT4 | 18 KiB | 4.5 GiB | **3.5 GiB** |

That is ~3.7x what the best weight-quant step (Q3_K_XL, 0.95 GiB) would give,
and it is a better trade for this workload for a reason worth stating plainly:

> Weight quantisation degrades EVERYTHING -- world knowledge, tool-call
> precision, code syntax. KV quantisation degrades ONE thing: long-range
> retrieval fidelity.

Keys decide *where* attention goes; values only *what it carries*. So K and V
are configurable independently and `K FP8 / V INT4` is the conservative step.

`--kv-cache fp8 | v4 | int4`.

## Left to do

1. ~~**Vision, toggleable.**~~ DONE -- see reports/PHASE_9.md.
2. **A GGUF loader in `model.cu`.** The container, all 13 dequantisers and the
   fused GEMV are in and gated; what is missing is the path that maps GGUF
   tensor names onto `LayerWeights` and dispatches the linears to `gguf_gemv`
   instead of the AWQ GEMV. Until that exists no GGUF checkpoint runs end to
   end, which is why there is no GGUF row in reports/QUANT_ACCURACY.md and why
   "AWQ and GGUF are viable substitutes" cannot yet be claimed. The kernel side
   is at a composition-weighted 355 GB/s, about 61% of llama.cpp; the next lever
   there is moving the IQ4_XS/IQ3_S codebook lookups into shared memory, since
   those two types are two thirds of UD-Q3_K_XL by bytes.
3. **RoPE extension past 262144.** The real prize of a smaller quant: more KV
   headroom is only useful if the positions above 262144 are usable, which needs
   YaRN-style extrapolation. Quality there has to be measured, not assumed.
4. **Drafter state in the prefix snapshot.** The DFlash2 drafter's own 42 MB
   context cache is not snapshotted, so acceptance dips briefly after a cache
   hit.
5. **G2**: 64K decode is 82% of 4K against an 85% bar.

## The quant decision, and the experiment that is deliberately deferred

**UD-IQ4_XS was chosen and then MEASURED, and the measurement killed it.** The
published file sizes are misleading, because they include the embedding and the
output head. Summing actual tensor bytes:

| | GGUF UD-IQ4_XS | ours (AWQ INT4 g128) |
|---|---|---|
| body (what decode reads every token) | **11.941 GiB** | **11.859 GiB** |
| output head | 0.814 GiB (Q5_K) | 1.203 GiB (INT8) |
| token embedding | 0.509 GiB (Q3_K) | 0 -- host resident, bf16 |
| resident total | 12.755 GiB | 13.062 GiB |

**The IQ4_XS body is 0.08 GiB LARGER than ours.** IQ4_XS is 4.25 bits/weight and
our AWQ INT4 g128 is 4.16, so they are the same bit budget; the file-size
advantage was never in the body. The entire 0.31 GiB net saving comes from the
head being Q5_K rather than INT8 -- about 10k tokens of context -- and we would
get the same by quantising our own head lower, with no second backend at all.

So UD-IQ4_XS is not worth adopting. The container and dequantisation work is NOT
wasted: it is exactly what is needed to evaluate or adopt any GGUF, and it is
committed and gated bit-exact against ggml.

The real headroom is further down the ladder. Everything below it trades measurably more error:
Unsloth's published KLD on the same size class is 0.024 for Q4_K_XL, 0.081 for
Q3_K_XL (3.4x) and 0.221 for Q2_K_XL (9.2x).

Agentic workloads are the *worst* case for quantisation, not the best, and that
is why the bar is set here rather than lower:

* Errors compound multiplicatively across a 40-step tool chain.
* Multi-hop retrieval over a long context degrades before fluency does —
  quantisation noise raises the floor on attention logits.
* Code and tool JSON are low-entropy: perplexity averages over exactly the
  confident tokens whose flipping breaks a bracket or a field name.
* World knowledge goes first, because rare-token logits ride on a few large
  weights. Measured on this model: INT4 `lm_head` gives KL 1.8e-2 against INT8's
  7.8e-4, which is why the head is INT8.

### DONE: KL against true BF16, measured on this box

This was the gap: our 6.99e-4 KL was against a BF16 model **dequantised from the
INT4 checkpoint**, so it quantified *kernel fidelity*, not AWQ's quantisation
loss, and it was compared against Unsloth's table for a different model. That is
now measured directly — see `reports/QUANT_ACCURACY.md`.

The `llama-perplexity --kl-divergence` route sketched here was not the one taken.
It only compares GGUF against GGUF inside llama.cpp, which cannot say anything
about our AWQ path or about EXL3, and it scores wikitext-shaped text. Instead
`tools/quantcmp_*` runs the real `Qwen/Qwen3.8-27B` BF16 through HF with layer
offload, teacher-forces a prompt set built from agentic-coding workloads, and
scores any engine that can emit fp16 logits for the same token ids:

| | KL mean | KL p99 | top-1 |
|---|---:|---:|---:|
| ours, AWQ INT4 + INT8 head | 1.495e-01 | 4.263 | 98.56% |
| EXL3 3.00bpw | 1.672e-01 | 5.216 | 97.74% |
| EXL3 3.50bpw (VRAM-matched) | **1.308e-01** | **3.493** | 98.51% |

The headline finding is the *shape*: median KL 5e-05, p99 4.26. Quantisation
error concentrates on a few positions and is invisible in an average, so any
future quant decision on this repo should be made on the tail, not the mean.

**Still open: a GGUF row.** It needs a GGUF loader in `model.cu`, which does not
exist — the container, dequantiser and fused GEMV are all gated in isolation but
nothing runs end to end. Comparing llama.cpp against a BF16 GGUF instead would
measure llama.cpp, not us, so the row is left empty rather than faked.

## DFlash2 transfers to any quant, unchanged

Worth recording because it removes a whole workstream. The drafter consumes only

* target hidden states at layers [5, 19, 33, 47, 61] — activations, bf16,
  independent of how the weights are stored,
* the target's embedding table, gathered to bf16 whatever its storage,
* the target's `lm_head`, through `model_apply_head`, which already dispatches
  over INT4 / INT8 / BF16.

Nothing in it touches the target's weight format, so **the same DFlash2
checkpoint drives a GGUF target with no changes and no second drafter**. The one
measurable risk is that heavier target quantisation moves the hidden states away
from what the drafter was trained against, lowering acceptance — which shows up
directly as mean-accepted-per-round in `bench_dflash`, so it is a measurement
rather than a guess.
