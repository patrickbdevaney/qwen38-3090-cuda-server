# PHASE 6 — Prefix caching for a hybrid GDN model

Status: **90.5× faster prefill on the second turn of a conversation**, measured
end to end through the server. G8's bar was 20×.

| | measured |
|---|---|
| turn 1, 6635-token prompt, cold | 689.7 tok/s prefill (9.6 s) |
| turn 2, 6736-token prompt, 6715 cached | 62448 tok/s effective (**90.5×**) |
| gate_prefix, 3000-token prompt | 46.4× |
| snapshot cost | 150 MiB pinned **host** memory per slot, **0 device** |
| restore | 21 ms |
| KL(cold ‖ warm) at the reuse point | 3.75e-04 |

---

## 1. Why this is not just a KV cache

A KV cache can be truncated to a prefix. Position *p*'s keys and values do not
depend on anything after *p*, so reusing `[0, p)` is a matter of not overwriting
it. **A GatedDeltaNet layer's recurrent state cannot be truncated.** It is
updated in place, `S_t` is a function of `S_{t-1}`, and there is no way to run
the recurrence backwards to recover the state at an earlier position.

Everyone serving these models hits this. vLLM tracks it as an open issue and
ships `--mamba-cache-mode align` as experimental; SGLang built a
`MambaRadixCache`; LMCache reinterprets the state as an opaque page so its
transfer path does not need to understand it.

So a prefix is reusable only at positions where the state was **snapshotted**,
and the state is all-or-nothing: either you have it at exactly position *p* or
you do not.

## 2. Where to snapshot

The obvious answer — checkpoint every N tokens, like a KV block table — is
almost pure waste. Marconi (arXiv 2411.19379) measured that at 32-token blocks,
**25% of KV blocks get reused by a later request but only 0.4% of SSM states do**,
a 65× gap. Recurrent states are large and rarely land on a boundary anyone comes
back to.

Their policy is to admit at most a couple of states per sequence: at points where
the request tree branches, and **at the last decoded token**. The second is the
one that matters here, because "the last decoded token" is exactly where a chat
turn ends, and turn N+1's prompt is turn N's context plus new user text. On
SWE-Bench agent traces that single admission point is worth 34.4× on hit rate and
36–71% of P95 TTFT.

This implementation snapshots in two places: at the end of prefill (the branch
point for a prompt that gets re-sent or extended without the reply) and at the
last token that went through the model (the conversation-resume point).

## 3. Where this differs from Marconi: host memory

Marconi keeps snapshots in GPU memory because it serves many sequences
concurrently and has no choice. We serve one.

Our state is 48 GDN layers × 48 value heads × 128 × 128 = 37.7 M floats =
**144 MiB in fp32**, plus 6 MiB of convolution state. Putting four of those on
the GPU would cost 600 MiB — directly out of the KV budget, which is the thing
we are trying to grow.

Over PCIe from pinned host memory, 150 MiB moves in **21 ms**, against a prefill
that costs seconds. So snapshots live in host RAM: slots cost host memory, which
we have 60 GB of, and **zero device memory**. Prefix caching and a maximal
context stop competing.

The snapshot also carries the 0.5 MB logit vector for its last position, which
makes the state genuinely complete — a prompt that is exactly a cached prefix
needs no prefill at all, not even one token.

## 4. Three exactness claims, and they are not equal

The first version of this gate asserted one thing: a warm request must be
byte-identical to a cold one. It failed, and chasing it turned up something
worth knowing on its own.

**Chunked prefill is not chunk-invariant.** Prefilling 1024 tokens in one call
versus two gives different results:

```
N=1024, one chunk vs:
  chunk  512 : 215979 / 248320 logit bits differ, max|d| 3.44e-01
  chunk  256 : 190020 / 248320                    max|d| 9.38e-02
  chunk   64 : 186593 / 248320                    max|d| 1.02e-01
```

That is not a state-carry bug — the recurrent state is carried correctly — it is
that `lin_path` picks a different kernel for different T, cuBLAS picks different
algorithms for different M, and the attention prefill tiles differently. PHASE_3
predicted the amplification: the GDN delta rule computes a residual, so rounding
differences grow rather than damp.

So the gate now makes three separate claims:

| | claim | measured |
|---|---|---|
| **A** | store + restore is a bit-for-bit fp32 round trip | 24/24 tokens identical — **exact** |
| **B** | resuming on a prefill **chunk boundary** is exact, because the remaining chunks have identical shapes | **0** of 248320 logit bits differ — **exact** |
| **C** | resuming after **generated** tokens is not exact and cannot be | top-1 agrees, KL 3.75e-04 |

C cannot be exact because that state was produced by D single-token decode steps
while a cold run reaches the same position with a large chunked prefill —
different kernels, different summation orders. Claiming otherwise would be false.
What can be said is that 3.75e-04 is **smaller than the 6.99e-04 this model's
INT4 weights already cost** against the bf16 reference, so the cache sits well
inside a numerical envelope we had already accepted.

C is also the only mode that can reuse generated tokens, and therefore the only
one that reaches the 20× bar. B is available for callers who need run-to-run
bitwise reproducibility more than they need the last few hundred tokens of reuse.

## 5. The bug this exposed: the cuBLAS cliff, again

The first working version reached only 14.4×. A 25-token warm prefill was taking
**286 ms**.

`lin_path` sent anything with `T > 16` to cuBLAS, and cuBLAS dequantises every
weight in the model to bf16 into a workspace before it multiplies — a fixed cost
we had already measured at **12.55× a decode step at T=8** in PHASE_7. The
tensor-core path takes at most 16 rows per launch, but it is weight-stream bound
and nearly flat in M, so chunking it by 16 costs one pass over the weights per
chunk and nothing else:

```
MMA, chunked by 16 :  ceil(T/16) * 45 ms
cuBLAS             :  ~250 ms + T/1440 s
```

which cross at about T = 112. The threshold is now 128 (`QWEN_MMA_MAX_T`), and
the 25-token warm prefill went **286 ms → 92 ms**, taking gate_prefix from 14.4×
to 46.4×.

This also moved `gate_forward`, which had been the one failing gate at 99.48%
against its own 99.5% bar. It now passes at 99.74%. Honesty demands the detail:
strict top-1 went *down* (382/384 → 381/384) while exact bf16 ties went up
(0 → 2), so the tie-inclusive metric crossed the bar. That is near-ties landing
differently, not an accuracy improvement, and it should not be read as one.

## 6. Correctness of reuse, end to end

The server test is a 300-line project-notes system prompt plus a question, then a
follow-up:

```
turn 1: prompt_n=6635  cached_n=0     prefill=689.7 tok/s    "There are 300 modules"
turn 2: prompt_n=6736  cached_n=6715  prefill=62447.8 tok/s  "Module 7 depends on Module 6"
```

Turn 2 answers a question that can only be answered from the document it never
re-read. That is the check that matters: the restored state carries the content,
not just the shape.

## 7. Not done

* **The drafter's context cache is not part of the snapshot.** On a cache hit the
  DFlash2 drafter restarts its own 2048-row context from wherever the prefill
  resumed. Every drafted token is still verified, so this is a throughput dip and
  never a correctness problem, but acceptance sags for a while after a hit. The
  drafter's KV is only ~42 MB and belongs in the snapshot; that is the obvious
  next improvement.
* **Eviction is plain LRU.** Marconi needs a FLOP-efficiency score because its
  snapshots compete with the KV cache for device memory. Ours compete with
  nothing, so LRU is adequate until slots become scarce.
* **No radix tree.** Slots are compared by longest common prefix, which is
  correct and O(slots × length) — fine at four slots, wrong at four hundred.
