# qwen38-3090-cuda-server

Pure CUDA/C++ single-node inference server for **Qwen3.8-27B** (AWQ INT4 W4A16) with a
**DFlash2 block-diffusion drafter**, targeting one specific card: **RTX 3090 (GA102, sm_86,
24 GB GDDR6X)**. No Python in the serving path. No batching, no vision, single sequence.

**Status: Phase 0 complete. Gate 0 passed. No kernels written yet.** See
[`reports/PHASE_0.md`](reports/PHASE_0.md).

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
| vLLM W4A16 | not yet measured | — |
| this project | not yet built | — |

llama.cpp reaches **77% of measured DRAM bandwidth** — it is a strong baseline, not a soft one.
Our expected advantage is mostly the smaller weight footprint (12.435 vs ~17.0 GiB/token), not
kernel heroics.

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
bench/microbench.cu        Phase 0 hardware probe (DRAM BW, BF16 MMA, cp.async, smem ceiling)
bench/bench_openai.py      benchmark harness, used identically for every server
bench/prompt_suite.json    workload basket: math, code x2, prose, toolish
tools/inspect_model.py     GATE 0 - derives every shape from the checkpoint, asserts the arch
tools/vram_budget.py       the budget contract src/main.cpp must enforce at startup
reports/                   PHASE_0, PRIOR_ART, QUANT_CHOICE, BASELINES + raw logs
src/                       empty until Phase 1; kernels are forbidden before Gate 0 passes
```

## Reproducing Phase 0

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
4. Speculative decoding is **lossless**: greedy output with speculation on must be token-for-token
   identical to speculation off. There is no `tau` in DFlash2 — the official reference has no
   acceptance threshold of any kind.
