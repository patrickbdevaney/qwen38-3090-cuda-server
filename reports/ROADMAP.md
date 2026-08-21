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

## Left to do

1. **Vision, toggleable.** 27-block SigLIP-style ViT, 0.858 GiB. Costs 28K
   tokens of context when enabled, which is why it is a launch flag.
2. **A second quant backend on UD-IQ4_XS.** See below.
3. **RoPE extension past 262144.** The real prize of a smaller quant: more KV
   headroom is only useful if the positions above 262144 are usable, which needs
   YaRN-style extrapolation. Quality there has to be measured, not assumed.
4. **Drafter state in the prefix snapshot.** The DFlash2 drafter's own 42 MB
   context cache is not snapshotted, so acceptance dips briefly after a cache
   hit.
5. **G2**: 64K decode is 82% of 4K against an 85% bar.

## The quant decision, and the experiment that is deliberately deferred

**Chosen: UD-IQ4_XS** (14.3 GB). It is the only step below our AWQ INT4 g128
stack (15.30 GB) that stays in the same fidelity class, and the ~1 GB it frees is
what vision + 262K needs. Everything below it trades measurably more error:
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

### FUTURE OPTION, not a blocker: measure KLD per quant on this model

There is a real gap in the argument above. Our measured 6.99e-4 KL is against a
BF16 model **dequantised from the INT4 checkpoint**, so it quantifies *kernel
fidelity*, not AWQ's quantisation loss. The table it is compared against is
Unsloth's, measured on a different model. Nobody has measured this model's
quants against true BF16 on this box.

`llama.cpp` ships the tool for it:

```bash
# once: reference logits from the F16 GGUF (~54 GB on disk, partial offload)
llama-perplexity -m Qwen3.8-27B-F16.gguf -f corpus.txt --kl-divergence-base base.dat
# then per candidate
llama-perplexity -m Qwen3.8-27B-UD-IQ4_XS.gguf -f corpus.txt --kl-divergence-base base.dat --kl-divergence
```

That yields per-quant KL divergence **and top-1 agreement on this model**, for
IQ4_XS / Q3_K_XL / Q2_K_XL side by side. Corpus should be the workload, not
wikitext: agent transcripts, tool-call JSON, and repository code.

**This is deliberately deferred.** Its value is finding whether a *lower* quant
is viable on our actual workload and therefore buys headroom for RoPE-extended
context — which is an optimisation on top of a finished server, not a
prerequisite for one. Implementing GGUF, vision and the rest comes first; the
measurement tells us how far down the ladder we are allowed to go afterwards.

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
