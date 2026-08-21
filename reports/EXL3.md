# EXL3 on this box — measured, not projected

The open question from Phase 9 was whether some *other* RTX 3090-viable
quantisation buys KV headroom that AWQ INT4 does not, at decode speed that is at
least a wash. EXL3 (exllamav3's QTIP-derived trellis format) was the leading
candidate: it is the only widely-used format with a genuinely different
quantiser, and turboderp publishes Qwen3.8-27B at 2.50 through 5.00 bpw.

This report is the measurement. Short version: **EXL3 3.00bpw decodes ~4% faster
than our AWQ path at short context, converges to identical by 32K, and buys no
VRAM headroom at all once the KV cache is counted.**

## Getting it to run

exllamav3 1.4.2 does not build against CUDA 12.1 (host compilation of the
`.cpp` files parses `__device__` constructor bodies in `util.cuh` and trips over
`__halves2half2`). `sudo` is not available on this box, so the toolkit came from
a conda prefix instead of a system install:

```bash
conda create -y -p ~/qwen38-weights/cuda128 -c nvidia \
  cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8 cuda-nvrtc-dev=12.8 \
  libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev libcufft-dev
```

Conda lays headers and libs out under `targets/x86_64-linux/{include,lib}`,
which `CUDA_HOME`-based build systems do not look at, so those have to be linked
into `$CUDA_HOME/{include,lib64}`. With that, exllamav3 builds clean and
unpatched. **The 12.1 patch was the wrong fix; the toolkit was.**

### The crash, and what it actually was

With the extension built, every run died with

```
GPU assert: an illegal memory access was encountered .../quant/coop_autotune.cu 464
```

which is a red herring — that line is the next runtime API call that observes a
sticky error, not the fault. `CUDA_LAUNCH_BLOCKING=1` moved the report to
`gdn.cu:1448`, and compute-sanitizer named it outright:

```
Invalid __global__ read of size 2 bytes
  at conv1d_update_kernel<(bool)1, (bool)0>+0x6e0 in gdn.cu:1309
  and is 135,876,893,999,102 bytes before the nearest allocation
```

`gdn.cu:1309` is `old_state[k] = state_d[k]`, i.e. reading the GatedDeltaNet
conv state. Printing the tensor at the call site gave the answer:

```
CONV cs ((1, 10240, 4), device(type='meta'))  | slots (cuda:0, [0])
```

**`conv_state` was on the `meta` device** — shape but no storage. The cause is
API ordering, not a library bug: `Cache(model, ...)` must be constructed
*before* `model.load()`, because loading is what materialises the recurrent
state tensors. Building the cache afterwards leaves every GDN layer's conv and
recurrent state on `meta` and the kernel dereferences a null base pointer.
This is documented in the `Cache` docstring and is easy to get backwards.

Worth recording for our own code: a fault this far from its cause cost several
hours of bisection. Our gates check numerics against a reference in the same
commit precisely so this class of thing surfaces at the kernel, not four
subsystems downstream.

## Decode, matched cache size

Both rows are a 33408-token cache on the same idle GPU, greedy, 128 generated
tokens, decode timed per token with prefill excluded. EXL3 numbers come from
`scratchpad/exl3_decode.py` driving the exllamav3 `Generator` streaming
iterator with a unique prompt per context so its prefix cache cannot skew the
result; ours from `./build/bench_decode <model> 33408 8 1 4096 1 0`.

| ctx | ours, AWQ INT4 + FP8 KV | EXL3 3.00bpw + FP16 KV | EXL3 vs ours |
|---:|---:|---:|---:|
| 4096 | 45.5 tok/s | **47.50** | +4.4% |
| 16384 | 42.8 | **44.83** | +4.7% |
| 32768 | 40.2 | **40.52** | +0.8% |
| resident | 16585 MiB | 16352 MiB | −233 MiB |

For reference on the same box: llama.cpp UD-Q3_K_XL measures 44.63 tok/s.

EXL3's trellis kernels are genuinely good — a few percent ahead of our fused
Marlin-class W4A16 GEMV at short context is a real result, not noise. But the
advantage decays as context grows, because past a few thousand tokens the
bandwidth bill is the KV cache, not the weights, and that is where the two
formats diverge in the direction that matters to us.

## Why it buys no headroom

| | weights resident | KV per token |
|---|---:|---:|
| ours: AWQ INT4 body, 8-bit lm_head, INT8 embed | 14.25 GiB | 32 KiB (FP8) / **18 KiB (INT4)** |
| EXL3 3.00bpw | 12.87 GiB | **67 KiB (FP16)**, measured |

EXL3 is 1.38 GiB lighter in weights and 35 KiB/token heavier in cache, so it is
ahead only below ~40K tokens and falls behind everywhere past that. The KV
figure is measured, not assumed: growing the cache from 8192 to 33408 tokens
moved resident VRAM by 1.62 GiB, which is 67 KiB/token.

Extrapolated to the context length this server actually targets — and this is a
`PROJECTED:` line, not a measurement, because it does not fit and therefore
cannot be run:

```
PROJECTED: EXL3 3.00bpw at 262144 ctx = 12.87 GiB weights + 16.8 GiB FP16 KV
           = ~29.7 GiB, which does not fit in 24 GiB.
```

Ours at 262144 with INT4 KV measures **19749 MiB**, with 4.4 GiB to spare.

exllamav3 does ship a quantised cache (`CacheLayer_quant`, `k_bits`/`v_bits`),
which would close most of that gap; measuring it is still open — the attempt
failed with `stloader: failed to allocate pinned pool` while 50 GiB of page
cache from the BF16 download was resident, and is queued behind that download.

## Also worth knowing

The published EXL3 sizes make the ceiling clear before any kernel work:

| EXL3 | total | vs our AWQ body+heads (14.25 GiB) |
|---|---:|---:|
| 2.50bpw | 11.45 GiB | −2.80 |
| 3.00bpw | 12.87 GiB | −1.38 |
| 3.50bpw | 14.29 GiB | +0.04 |
| 4.00bpw | 15.70 GiB | +1.45 |
| 5.00bpw | 18.53 GiB | +4.28 |

Only 2.50 and 3.00 bpw are smaller than what we already run, and 2.50bpw is
below the bit rate where we would want to trust a coding agent without a KL
measurement against BF16.

## Where this leaves the "build an EXL3 backend" question

The original condition was: *if exllama remains accurate and fastest, build a
kernel server for it too*. It is not fastest by a margin that would justify a
second trellis-decoder backend — +4% at 4K, +0.8% at 32K, behind at the context
lengths this server exists for. **Accuracy is the only remaining reason to keep
it in scope**, and that is exactly what the BF16 comparison now downloading will
settle. Quality on this box is at least sane: 3.00bpw answers factual, coding
and arithmetic prompts coherently and correctly (`scratchpad/exl3_qual.py`).

Decision deferred to the BF16 KL numbers. If EXL3 3.00bpw is materially closer
to BF16 than AWQ INT4 is, it earns a place as a weight option for users who
want it. If it is a wash, it does not, and the effort belongs in the GGUF GEMV
instead — that one has a clear 546 GB/s target and an existing 347 GB/s.
