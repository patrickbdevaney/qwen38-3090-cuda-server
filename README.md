# qwen38-3090-cuda-server

Pure CUDA/C++ single-node inference server for **Qwen3.8-27B** (AWQ INT4 W4A16) with a
**DFlash2 block-diffusion drafter**, targeting one specific card: **RTX 3090 (GA102, sm_86,
24 GB GDDR6X)**. No Python in the serving path. No batching, no vision, single sequence.

**Status: running.** Autoregressive decode, DFlash2 speculative decode, FP8 KV cache, CUDA
graphs, and an OpenAI-compatible server. Measured on this box, from committed binaries:

| | measured |
|---|---|
| autoregressive decode, 4K ctx | **45.8 tok/s** |
| llama.cpp Q4_K_M, same box, same prompts | 38.41 tok/s |
| **DFlash2 speculative decode, mean of 3 prompts** | **100.1 tok/s (2.32x AR)** |
| DFlash2 through the server, greedy request | **133.3 tok/s**, 7.54 accepted per round |
| **prefix cache, 2nd turn of a conversation** | **90.5x faster prefill** (6715 of 6736 tokens reused) |
| mean accepted tokens per block of 8 | 4.10 / 5.79 / 6.83 |
| decode GEMV, traffic-weighted | 769.8 GB/s = 84.2% of measured DRAM |
| prefill GEMM | 70.1 TFLOPS = 86% of measured BF16 peak |

Full tables and every invocation: [`reports/BENCHMARKS.md`](reports/BENCHMARKS.md).
Phase reports: [`PHASE_0`](reports/PHASE_0.md) .. [`PHASE_7`](reports/PHASE_7.md).

Peak VRAM at 128K context is **21.47 GB** of 24, so the full 128K window fits with the model,
the INT8 head and the INT8 embedding resident.

All 16 gates pass. Known misses, stated as misses: 64K decode is 82% of 4K against an 85%
bar, and prose speculation lands below its 120 tok/s bar.

---

## Why this model on this card

Qwen3.8-27B is a hybrid: **48 GatedDeltaNet (linear attention) layers and 16 full-attention
layers**, in a strict `16 x (L, L, L, A)` pattern. Only the 16 attention layers hold a KV cache.

Verified against the checkpoint (`tools/inspect_model.py`):

```
16 attn layers x 4 KV heads x 256 head_dim x 2 (K and V) = 32,768 elements/token
```

| dtype | per token | 32K | 128K | 262K |
|---|---|---|---|---|
| BF16 | 64 KiB | 2.00 GiB | 8.00 GiB | 16.00 GiB |
| **FP8 e4m3** | **32 KiB** | 1.00 GiB | **4.00 GiB** | 8.20 GiB |

A dense 27B model would need 4x that. **This is why 128K context fits on a 24 GB card alongside
the full model and a 2B drafter**, and it is a direct consequence of the 3:1 GDN-to-attention
layout. The GatedDeltaNet recurrent state is 144 MiB in fp32 and **constant in context length**.

## Measured VRAM budget

From real tensor sizes, not estimates (`tools/vram_budget.py`). Display attached; 22.64 GiB free.

| Component | Size |
|---|---|
| LM body INT4 g128 (+ scales, zeros, norms) | 11.820 GiB |
| `lm_head` INT4 g128 (repacked from BF16) | 0.615 GiB |
| `embed_tokens` INT8 rowwise (repacked) | 1.185 GiB |
| DFlash2 drafter (INT8) | 1.792 GiB |
| GDN recurrent + conv state (fp32) | 0.146 GiB |
| GDN rollback buffers (2x) | 0.292 GiB |
| GDN prefix-cache snapshots (8 x bf16) | 0.584 GiB |
| Activations + prefill workspace | 1.400 GiB |
| **Fixed total** | **17.835 GiB** |
| **Remaining for KV** (arena 22.140) | **4.305 GiB** → **141,074 tokens at FP8** |

Both checkpoints ship `lm_head` **and** `embed_tokens` as BF16 (2.368 GiB each). Repacking them
reclaims **2.94 GiB** — the difference between 128K and ~35K of context. Both repacks are gated
on KL divergence, not assumed free.

## Measured hardware (RTX 3090, sm_86)

`bench/microbench.cu` → [`reports/logs/microbench_sm86.log`](reports/logs/microbench_sm86.log)

| | measured |
|---|---|
| DRAM streaming read | **914.2 GB/s** (97.7% of 936.1 theoretical) |
| BF16 `mma.sync.m16n8k16` | 81.6 TFLOPS (at the 1965 MHz boost the card actually runs) |
| L2-resident read | 8603 GB/s |
| Shared memory / block | **99 KB opt-in** |
| Arithmetic-intensity balance point | 89.3 FLOP/byte |

**Decode roofline:** 12.435 GiB of weights per token / 914.2 GB/s = 14.60 ms → **68.5 tok/s
autoregressive ceiling.**

## Baselines, measured on this box

[`reports/BASELINES.md`](reports/BASELINES.md). Same prompts, same sampling params, same harness.

| | decode tok/s | prefill tok/s |
|---|---|---|
| llama.cpp Q4_K_M, autoregressive | **38.41** | 1318 (pp8192) |
| llama.cpp Q4_K_M + MTP speculation | **51.30** median (82.94 math → 43.96 prose) | — |
| vLLM W4A16 | not measured — see below | — |
| **this project, autoregressive** | **45.8** | 1440 (projected from 70.1 TFLOPS at M=4096) |
| **this project, DFlash2 speculative** | **100.1 mean, 122.4 best** | — |

llama.cpp reaches **77% of measured DRAM bandwidth** — it is a strong baseline, not a soft one.
The advantage here is mostly the smaller weight footprint (12.435 vs ~17.0 GiB/token) plus the
84.2% aggregate the decode GEMV reaches, not kernel heroics.

vLLM was **not** benchmarked. It would have needed a from-source build on CUDA 12.8 to avoid a
driver upgrade that has previously broken suspend on this desktop. Its absence is stated rather
than filled in with an estimate.

**llama.cpp cannot run the DFlash2 drafter.** It auto-detects `draft-dflash` but its arch
definition is DFlash **1**: no two-tap dynamic convolution, no candidate selector, so it creates
58 of the GGUF's 81 tensors and aborts. Verified against `gguf-py` and
`gguf-py/gguf/constants.py:4849` on master `749f688`.

## What this server will not do

- **No continuous batching.** Single sequence plus one background prefill slot. Explicitly out
  of scope.
- **No vision.** v1 is text-only. The loader detects and skips `model.visual.*` (0.858 GiB) and
  the API returns 400 on image/video content parts. The hook is left in place.
- **No multi-GPU, no CPU offload.** One 3090.

## Layout

```
src/kernels/               gemv_w4a16 (decode + skinny GEMM + tensor-core W4A16), gdn,
                           attn (FP8 KV, split softmax), elementwise, sampling
src/model/                 loader, layer stack, CUDA graph capture, VRAM budget
src/draft/                 DFlash2 block-diffusion drafter and its candidate selector
src/spec/                  speculative loop, GDN state rollback, acceptance
src/cache/                 prefix cache: GDN state snapshots in pinned host memory
src/tokenizer/             BPE, NFC, Unicode tables, chat template
src/server/                OpenAI-compatible endpoints, SSE, tool calls, reasoning parser
tests/gate_*               one standalone executable each; `ctest` runs them all
bench/                     microbench (Phase 0 probe) plus per-kernel and end-to-end benches
tools/                     Python, ONE-TIME ONLY: reference dumps and checkpoint inspection
reports/                   PHASE_0..7, BENCHMARKS, PRIOR_ART, QUANT_CHOICE, BASELINES + logs
```

## Building and running

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DQWEN38_MODEL_DIR=/path/to/Qwen3.8-27B-W4A16-AWQ \
      -DQWEN38_DRAFT_DIR=/path/to/Qwen3.8-27B-DFlash2 \
      -DQWEN38_BF16_DIR=/path/to/bf16-reference     # gates only
cmake --build build -j
ctest --test-dir build            # every gate

./build/cuda_server  --model /path/to/Qwen3.8-27B-W4A16-AWQ --port 8080
./build/bench_decode /path/to/Qwen3.8-27B-W4A16-AWQ 131072 8 1
./build/bench_dflash /path/to/Qwen3.8-27B-W4A16-AWQ /path/to/Qwen3.8-27B-DFlash2 192 1
```

`QWEN_DEBUG_SYNC=1` synchronises after every stage of `run_layer` and aborts at the first
faulting kernel; `=2` turns the same hooks into a per-stage wall-clock profile.

### Reproducing Phase 0

```bash
nvcc -O3 -arch=sm_86 -o bench/microbench bench/microbench.cu && ./bench/microbench
python3 tools/inspect_model.py  /path/to/Qwen3.8-27B-W4A16-AWQ
python3 tools/vram_budget.py    /path/to/Qwen3.8-27B-W4A16-AWQ \
        --drafter-dir /path/to/Qwen3.8-27B-DFlash2 --ctx 131072
```

Model weights and the DFlash2 drafter are **not** in this repo. Fetch them from Hugging Face:
`philbert440/Qwen3.8-27B-W4A16-AWQ`, `z-lab/Qwen3.8-27B-DFlash2`.

## Ground rules

1. Never fabricate a benchmark. Every tok/s here comes from a committed binary, a committed
   invocation and a committed log. Projections are prefixed `PROJECTED:`.
2. Numerics before speed. Every kernel lands with a reference comparison in the same commit.
3. Phase gates are hard stops with a written report.
4. Speculative decoding is **lossless in its acceptance rule**: longest-prefix argmax equality
   plus the target's correction emits exactly what the target would have emitted. There is no
   `tau` in DFlash2 — the official reference has no acceptance threshold of any kind.

   What batched verification does *not* give you is bit-identical arithmetic. Verifying eight
   rows runs the tensor-core W4A16 path, the prefill attention kernel and the batched GDN scan;
   decoding one row runs the GEMV, the split-softmax decode kernel and the single-step scan.
   Both are valid and they do not agree to the last bit, so a near tie can flip. `gate_spec`
   measures this rather than asserting it: over 4 prompts x 192 tokens one prompt diverges,
   under **both** drafters, and at that position the two candidate logits differ by exactly one
   bf16 ulp. The gate fails hard if a divergence ever has a real logit gap.
