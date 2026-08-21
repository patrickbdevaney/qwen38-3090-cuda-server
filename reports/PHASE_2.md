# PHASE 2 — W4A16 kernels

Status: **GATE PASSED.** Decode GEMV at **84.2%** of measured DRAM bandwidth
(gate: 80%). Prefill GEMM at **84%** of measured tensor-core peak.

| | measured | implies |
|---|---|---|
| decode GEMV, aggregate | **769.8 GB/s = 84.2%** of 914.2 | 57.3 tok/s from the linear layers alone |
| prefill GEMM, M=4096 | **68.6 TFLOPS = 84%** of 81.6 | **1410 tok/s** at 8K vs llama.cpp's measured 1318 |
| GEMV numerics | 2.6e-6 worst relative error vs fp64 | — |
| GEMM numerics | 3.2e-3 worst relative error vs fp64 | the bf16 output rounding floor |

---

## 1. Decode GEMV — three rounds, from 42% to 84%

Each step was A/B measured on real checkpoint weights, not reasoned about.

### v1 — the obvious mapping: 42%
One warp per output row, lanes striding the input dimension, `x` staged in
shared memory. Correct on the first try (1e-7 relative), and slow.

### Round 1 — transpose the assignment
With a warp per row, **every lane wants a different activation**, so the shared
-memory reads were 8- to 16-way bank conflicted. v2 gives **each lane its own
output row** and walks the input dimension warp-wide. Now every activation read
is a warp-wide broadcast, `x` can live in L2 instead of shared memory (freeing
occupancy entirely), and no final warp reduction is needed because each lane
already holds a complete row sum.

The cost is that the 32 rows a warp owns must be adjacent in memory. **That is
what the repack is for, and the only reason decode needs one at all** — the
on-disk layout is otherwise already ideal.

### Round 2 — break the dependent FMA chain: 42% → 61.5%
A single accumulator serialised each group into a **128-deep chain of ~4-cycle
FMAs**. Eight independent accumulators fixed it.

This was the largest single win and **it was not what I predicted**. The kernel
is nominally memory bound; it was latency bound instead. Worth recording in the
ledger: on this card, at this arithmetic intensity, ILP mattered more than the
bandwidth story suggested.

### Round 3 — fill the machine: 61.5% → 84.2%
The split heuristic targeted 3 waves. At `out=17408` that is 544 warps over 82
SMs — **under 7 warps per SM**, far too few to hide DRAM latency. Retargeted to
8 waves; swept 2/3/4/6/8/12/16 and confirmed 8 is the knee (12 is within noise,
16 regresses). Also fused the two prologue kernels into one and skipped the
reduction entirely when `splits == 1`: at **448 GEMVs per decoded token** every
dependent launch costs 2–3 µs of dead GPU time, which is why the smallest
tensors were running at 25%.

### Dequant without int→float
bf16 `0x4300` is exactly 128.0 and its ulp at that exponent is 1, so
`(0x4300 | n)` **is** the value `128 + n` for `n ∈ [0,15]`. Four LOP3s turn one
u32 into four bf16×2 pairs and `__bfloat1622float2` widens each in a single
instruction: ~2.5 ops per weight against ~5.

The `+128` folds into the zero point for free. The stored `q` and `zp` nibbles
are *both* offset by 8, so the offsets cancel — `w = (q_stored − zp_stored)·s` —
and with `f = 128 + q_stored` the correction is a single per-group constant
`C = 128 + zp_stored`, applied against a per-group sum of `x` computed once.

### Fused projections
`q|k|v`, `gate|up`, and the GDN `in_proj_qkv|z` pair all share an input, so they
are concatenated at load into single GEMVs. This is an architectural win, not a
benchmark trick: it folds `k_proj` and `v_proj` — 1024×5120 tensors that run at
**32%** of peak because they cannot fill the GPU — into a 14336×5120 tensor that
runs at **79.8%**, and turns six launches into three.

```
op                        shape        xN   MB each      GB/s   %peak
gdn.in_proj_qkv|z    16384x5120        48      43.9     751.4   82.2%
gdn.out_proj          5120x6144        48      16.5     592.6   64.8%
attn.q|k|v           14336x5120        16      38.4     729.5   79.8%
attn.o_proj           5120x6144        16      16.5     603.0   66.0%
mlp.gate|up          34816x5120        64      93.3     819.7   89.7%
mlp.down             5120x17408        64      46.7     777.9   85.1%
```

### On the metric
Per-tensor efficiency is a **diagnostic, not the gate**. What sets tok/s is the
traffic-weighted aggregate over a real decode step: the mlp and qkv projections
are 84.8% of body traffic, `o_proj`/`gdn.out_proj` are 14.5%, and `k_proj` +
`v_proj` together are **0.69%**. So `gate_gemv` checks numerics only and
`bench_decode_gemv` is the bandwidth gate, reproducing the real per-token mix of
48 GDN layers plus 16 attention layers.

---

## 2. Prefill GEMM — cuBLAS is the right answer here

Prefill inverts the decode picture: 8K tokens is **399 TFLOP of GEMM against
11.9 GiB of weight traffic**. The arithmetic takes seconds; the weights take
14 ms. Only tensor-core efficiency matters.

So rather than hand-writing a Marlin-class fused-dequant GEMM, this dequantizes
each weight tile to bf16 into a workspace and calls cuBLAS — which the directive
explicitly permits for the prefill path. **The dequant costs 0.7%**: one extra
write plus one extra read of the weights, 55.7 ms against ~8 s of GEMM at 8K.
Hand-writing a GEMM to reclaim 0.7% while risking correctness would be the wrong
trade, and is exactly the "optimizing prefill before decode is correct" failure
mode the directive names.

Chunk size matters a lot, and settles the directive's 2048-vs-4096 question:

| M | TFLOPS | % of peak | projected 8K prefill |
|---|---|---|---|
| 1024 | 51.5 | 63% | 1059 tok/s |
| 2048 | 64.0 | 78% | 1315 tok/s |
| **4096** | **68.6** | **84%** | **1410 tok/s** |

**Chunked prefill uses 4096.**

### A note on llama.cpp's prefill
llama.cpp measures 1318 tok/s at pp8192, which is **64.1 TFLOPS effective —
79% of this card's bf16/FP32-accumulate peak**. That is implausibly high for an
FP32-accumulate kernel, so llama.cpp is almost certainly accumulating in **FP16**
(142 TFLOPS peak on GA102, where 64.1 would be a realistic 45%). We accumulate in
FP32. FP16 accumulation would roughly double the headroom and is recorded as a
**Phase 9 option behind a KL gate**, not taken silently.

---

## 3. What did not need doing

**No repack for the on-disk layout beyond the 32-row interleave.** The
compressed-tensors layout is already coalesced for a row-walking warp; the only
change is interleaving 32 rows so the lane-owns-a-row mapping works, plus moving
the zero point from its packed `[out/8][G]` u32 form to one byte per (row,
group). The 4-bit zp packing would save 0.09 GiB per token (0.7%) at the cost of
a shift+mask per group, and the zero-point stream is already the smallest of the
three.

---

## 4. Open items

1. **The two 5120×6144 shapes** (`attn.o_proj`, `gdn.out_proj`) sit at 65–67%
   while everything else is 80–90%. They are 14.5% of body traffic, so closing
   that gap is worth roughly **2 tok/s**. Phase 9 autotuning target.
2. **CUDA graphs are now clearly necessary, not just nice.** 448 GEMVs per token
   at 2–3 µs of launch latency each is 1.0–1.4 ms against a 14.6 ms budget —
   7–10% — before any other kernel is counted. Directive §8.8 already plans this;
   Phase 2 makes the number concrete.
3. **The lm_head KL gate** (moved here from Phase 7 in the Phase 1 report) still
   needs running: quantizing `lm_head` BF16→INT4 reclaims 1.753 GiB and is
   load-bearing for the 128K claim, and it is *also* the DFlash2 drafter's output
   head. It needs the end-to-end forward from Phase 5 to measure properly, so it
   runs at the Phase 5 gate.

## The single riskiest open question

Still the drafter, unchanged. But Phase 2 adds a second-order concern worth
naming: **the projected 57.3 tok/s is linear layers only.** It excludes the GDN
recurrent scan, attention, norms, RoPE and sampling. The roofline ceiling is 68.5
tok/s and the linear layers already consume 16.5 ms of a 14.6 ms ideal budget's
worth of traffic at 84% efficiency. Everything Phase 3 and Phase 4 add is pure
overhead against that. **G1's revised 45 tok/s minimum has less headroom than it
looks**, and the GDN scan — 48 of 64 layers — is where it will be won or lost.
