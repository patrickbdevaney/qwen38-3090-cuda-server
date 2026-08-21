# PHASE 3 — GatedDeltaNet

Status: **GATE PASSED.** All stages match the transformers reference at T = 1
(fresh and warm state), 8, 64 and 512.

| | measured |
|---|---|
| decode, 48 layers | **0.76 ms/token** = 4.6% of the 16.5 ms weight-traffic budget |
| decode state bandwidth | **642 GB/s** (70% of measured peak) |
| prefill, 48 layers, T=4096 | **630 ms** ≈ 22% on top of ~2.9 s of prefill GEMM |

---

## 1. The algebraic step that makes decode cheap

The recurrence as written needs **two passes over the state**: `delta` depends on
a full reduction over `k`, and the output depends on the *updated* `S`. Expanding
the last line removes the second dependency:

```
out = (g·S + k ⊗ delta)ᵀ q  =  g·(Sᵀq) + delta · (k·q)
```

So `out` never needs the updated state. One pass computes `A = Sᵀk`, `B = Sᵀq`
and the scalar `k·q`; the writeback then produces the new `S`. Keeping the tile
in registers across the two makes it **one read and one write** of the state,
which is what the 302 MB/token traffic figure assumes.

At 642 GB/s that lands at 0.76 ms per token across all 48 layers.

## 2. Three things that were wrong, in order of how much they cost

### The launch that "wasn't a launch bug"
`cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, 101376)` **fails
silently**: the 99 KB opt-in ceiling covers static *and* dynamic shared memory,
and this kernel already holds ~1.5 KB statically. The limit stayed at the 48 KB
default and every launch returned `invalid argument` — which reads like a
grid/block error and is not one. The code now queries the ceiling, subtracts the
static usage, and **checks the return status**.

### The single-threaded preamble: the whole prefill cost
The l2norm of `q` and `k` plus the scalar `k·q` was a loop on thread 0 — **384
dependent shared-memory reads at ~30 cycles each, ~11k cycles per step**, which
was essentially the entire measured step time. Moved onto warp 0 with shuffle
reductions.

### The 48-block ceiling
One block per head meant 48 blocks holding 64 KB of shared memory each: **one
block per SM, 8 warps of a possible 48**, and 34 of 82 SMs idle. But the
recurrence is sequential only in `k` — **every `v` is independent**. `A[v]`,
`delta[v]`, `out[v]` and the `S[·][v]` column all depend on `v` alone, and the
only shared quantity is the scalar `k·q`, which a block recomputes in ~128 flops.
Splitting `v` 8 ways gives 384 blocks of 8 KB and lets several share an SM.

Also: the shared stride is padded to `V+1`. Without it a thread reads
`S[k][v]` at `k·V + v`, so the bank is `(j·V + v) % 32` and threads in different
k-groups but the same `v` collide.

Sweep of (threads, v-per-block, timestep-batch), decode and prefill measured
separately — the two want different shapes, so both are instantiated and
dispatched on `T`.

## 3. Numerics: four places the reference's rounding had to be reproduced

Each of these was found by the gate failing, not by reading the source.

1. **`beta` is bf16.** The reference computes `b.sigmoid()` on a bf16 tensor, so
   beta carries 8 mantissa bits before being widened to fp32 inside the
   recurrence. Computing it in fp32 is *strictly more accurate* and put us
   3.7e-3 away — enough to flip an argmax over 64 layers. Rounded to match.
2. **`l2norm` runs in bf16**, before the fp32 cast, so the reduction result, the
   `rsqrt` and the product are all rounded. The `1/sqrt(d)` scale is applied in
   fp32, because the reference applies it *after* the cast.
3. **`g` is fp32** and must stay so — the reference is explicit that `A`
   overflows to `-inf` in fp16 otherwise.
4. **`softplus` switches to the identity above 20**, matching PyTorch's
   threshold. `g` is exponentiated afterwards, so this is not cosmetic.

`expf` replaced `__expf` throughout. It measured as no change, but the delta rule
computes `(v_t − g·Sᵀk)` — a *residual* — so cancellation amplifies input
rounding, and ~2 ulp is not obviously free there.

### The residual disagreement, and what it means for Phase 5
After all four fixes the recurrent state still differs from the reference by
**2–6e-3 relative**, at T=1 as much as at T=512, so it is not accumulation. It is
a chain of legitimate bf16 rounding differences — in the conv output, in the
normalized q/k — that the delta rule's cancellation amplifies. Matching the
conv's accumulation order changed nothing.

**Both implementations are equally correct; they simply round differently.** This
matters for the Phase 5 gate, which asks for a *token-exact* match against HF over
256 greedy tokens. That may not be achievable without reproducing PyTorch's
kernel internals bit for bit. **If it fails, the honest response is to say so with
evidence and gate on top-1 agreement plus KL divergence against a higher-precision
reference instead** — not to loosen the tolerance quietly and call it exact.

## 4. Why not the chunked parallel form

The reference has `torch_chunk_gated_delta_rule` precisely because a sequential
recurrence is slow, and the measurement agrees: prefill sits at **630 ms per
4096-token chunk**, ~22% on top of the GEMM, and it is **barrier-bound**, not
bandwidth-bound. Batching timesteps to amortise global latency moved it only
710 → 687 ms, which confirms the diagnosis: the cost is the two `__syncthreads`
per step in a 4096-step dependency chain.

The chunked form (WY representation / UT transform) would cut the barrier count
by the chunk factor and is the correct fix. It is **not** implemented, because:

- decode — the product — is already at 4.6% of budget and would not benefit;
- prefill's practical effect is TTFT, which the Phase 6 prefix cache addresses
  far more directly for agentic workloads;
- it is genuinely intricate and every line of it is a new way to be subtly wrong
  in the 48 layers that carry most of the model.

**The cost of not doing it is stated rather than hidden: prefill lands near
~1150 tok/s against llama.cpp's measured 1318.** It is the top Phase 9 item.

A cheaper win was taken instead: the prefill conv now runs one thread per
(channel, timestep) rather than walking T serially per channel, which cut a
4.4 ms stage to 1.9 ms.

## 5. Open items

1. **Chunked GDN prefill** — §4. Worth roughly +170 tok/s of prefill.
2. **Decode is at 70% of state bandwidth**, against 84% for the GEMV. The
   remaining gap is ~0.2 ms/token. Phase 9.
3. The Phase 5 token-exactness question in §3 is now the **most likely gate to
   fail**, and it will fail for a reason that is understood rather than mysterious.

## The single riskiest open question

Unchanged in substance but sharper: **the drafter, and now specifically whether
speculative decoding can be verified at all.** Blocking test 1 requires
speculation-on output to be token-for-token identical to speculation-off. That is
an *internal* consistency check — both paths are our own kernels — so bf16
rounding against PyTorch does not threaten it. But the GDN rollback replays the
recurrence over the accepted prefix, and §3 shows this recurrence amplifies
rounding through cancellation. **A replayed state and a directly-advanced state
must agree bitwise, or Blocking test 1 fails for numerical reasons rather than
logical ones.** That needs testing early in Phase 7, before any acceptance-rate
number is trusted.
