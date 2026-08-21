# BASELINES.md

Measured on this box. Same prompts, same sampling params, same timing code as everything else:
`bench/bench_openai.py` against `bench/prompt_suite.json`, greedy (`temperature=0`, `seed=0`),
`max_tokens=256`, 2 warmup + 5 measured, median reported with p95.

Box: RTX 3090 (sm_86, 82 SM, 23.57 GiB, display attached), Ryzen 9 5900X, CUDA 12.1,
driver 570.133.07. Measured DRAM streaming read 914.2 GB/s (`bench/microbench.cu`).

> Neither `laguna-s1-cuda-server` nor `deepseek-v4-flash-0731-cuda` ever ran llama.cpp or vLLM
> locally; both closed their head-to-head gates against published vendor numbers. These are
> therefore the first real local baselines for this model family.

---

## 1. llama.cpp — MEASURED

Build `749f688` (master, 2026-08-21), `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`.
Model `ggml-org/Qwen3.8-27B-GGUF` **Q4_K_M**, 17.66 GiB on device, 26.90 B params,
`-ngl 99 -fa 1 -c 16384 -np 1 --jinja`.

### 1.1 `llama-bench` (`reports/logs/baseline_llamacpp_bench.log`)

| test | t/s |
|---|---|
| pp512 (prefill) | **1356.54 ± 27.67** |
| pp8192 (prefill) | **1318.39 ± 2.64** |
| tg128 (decode) | **38.83 ± 0.05** |

### 1.2 Autoregressive, per workload (`reports/logs/baseline_llamacpp_ar.json`)

| case | class | decode tok/s (median) | p95 |
|---|---|---|---|
| gsm8k_math | math | 38.37 | 38.42 |
| humaneval_code | code | 38.28 | 38.30 |
| mbpp_code | code | 38.41 | 38.42 |
| prose_chat | prose | 38.48 | 38.50 |
| tool_style | toolish | 38.46 | 38.50 |
| **OVERALL** | | **38.41** | |

Flat across workloads, as expected for AR. Consistent with `llama-bench`'s 38.83.

**Achieved bandwidth:** ~17.0 GiB of weights read per token (17.66 GiB total less the embedding
row-lookup) x 38.41 tok/s = 653 GiB/s = **701 GB/s, i.e. 77% of the measured 914.2 GB/s.**
llama.cpp is not leaving much on the floor here. This is the efficiency number to beat, and it
is far above the ~50% that laguna's survey attributes to vLLM/SGLang.

### 1.3 Speculative decoding — MTP head (`reports/logs/baseline_llamacpp_spec_mtp.json`)

Draft model `mtp-Qwen3.8-27B-Q8_0.gguf` (3.16 GB), `-ngld 99 --spec-draft-n-max 8`.
Peak VRAM 22,740 MiB.

| case | class | decode tok/s | vs AR |
|---|---|---|---|
| gsm8k_math | math | **82.94** | 2.16x |
| tool_style | toolish | 55.29 | 1.44x |
| humaneval_code | code | 51.30 | 1.34x |
| mbpp_code | code | 48.79 | 1.27x |
| prose_chat | prose | 43.96 | 1.14x |
| **OVERALL** | | **51.30** | **1.34x** |

The 2x spread between math (82.94) and prose (43.96) is exactly why the directive's workload
basket matters and why a single speculative tok/s figure is not a summary.

### 1.4 llama.cpp cannot run the DFlash2 drafter — VERIFIED

llama.cpp *does* auto-detect `draft-dflash` from GGUF metadata, but loading
`z-lab/Qwen3.8-27B-DFlash2-GGUF` (either BF16 or Q8_0) fails:

```
common_specu: auto-detected speculative type 'draft-dflash' from the draft model metadata
llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 81, got 58
```

The GGUF is **not** at fault — reading it with `gguf-py` shows all **81** tensors present,
including `blk.N.attn_conv_base`, `blk.N.attn_conv_proj.weight`, `blk.N.ffn_conv_base`,
`blk.N.ffn_conv_proj.weight` (5 each = 20) and `selector_hidden.weight`,
`selector_predecessor.weight`, `selector_successor.weight` (3). **20 + 3 = 23 = 81 − 58.**

`gguf-py/gguf/constants.py:4849` confirms it: `MODEL_ARCH.DFLASH`'s tensor list contains no
`conv_base`/`conv_proj` and no selector entries. **This build implements DFlash 1 — no two-tap
dynamic convolution, no candidate selector — and therefore cannot run DFlash2 for this model.**

So llama.cpp's best available speculative configuration on this box is the MTP head at
**51.30 tok/s**, and that is the number to beat.

---

## 2. vLLM — BLOCKED on the NVIDIA driver, not yet measured

**vLLM cannot run on this box as configured.** The constraint is exact and was verified, not
guessed:

1. Support for this model (`Qwen3_5ForConditionalGeneration`) is present in vLLM 0.27.1, along
   with `Qwen3_5MTP` and — notably — **`DFlashDraftModel`**. So vLLM *can* be a
   with-and-without-DFlash2 baseline, which llama.cpp cannot (§1.4).
2. Every vLLM release recent enough to have that support pins `torch==2.11.0` or newer, and the
   PyPI wheels for those are built for **CUDA 13**:
   ```
   ImportError: libcudart.so.13: cannot open shared object file: No such file or directory
   ```
   Reproduced on both 0.27.1 and 0.26.0. Forcing `torch==2.11.0+cu128` fixes torch itself
   (`torch 2.11.0+cu128 cuda 12.8 avail True, dev NVIDIA GeForce RTX 3090`) but not vLLM's own
   compiled extension `vllm._C_stable_libtorch`, which is the wheel's CUDA-13 half.
3. **CUDA 13 requires NVIDIA driver >= 580.** This box has **570.133.07** (CUDA 12.8).
   GeForce parts have no CUDA forward-compatibility package, so there is no way around it.
4. There is no CUDA-12 vLLM wheel index (`wheels.vllm.ai/cu128` → 404).
5. Older vLLM releases that would run on CUDA 12 (0.16.0, torch 2.9.1) predate this model
   architecture entirely.

The host is **Ubuntu 24.10, which is EOL**, and its archive has no `nvidia-driver-580`
(`apt-cache policy` shows 570.133.07 as both Installed and Candidate). Getting to 580 means the
graphics-drivers PPA or NVIDIA's `.run` installer on an out-of-support release, on a machine
reached over RustDesk — i.e. a real risk of losing the display and remote access.

### Decision: vLLM is out of scope

The operator's call, and it is a reasonable one: driver 580 risks a known suspend/resume
regression on this desktop, and a from-source CUDA 12.8 build is an hour of work for a number
that is only a comparison point. **llama.cpp is the measured competitor for this project.**

Consequences, stated so they are not forgotten:

- Directive §12 asks for a measured comparison against **both** llama.cpp and vLLM. That item is
  **not met and will not be**, unless the driver situation changes. `BENCHMARKS.md` must say so
  explicitly rather than quietly listing one baseline.
- vLLM would have been the *only* available with-and-without-**DFlash2** comparison, since
  llama.cpp cannot run DFlash2 at all (§1.4). So there is no external DFlash2 reference on this
  box. Our speculative numbers will be compared against **our own AR path** and against
  llama.cpp's MTP speculation, and the acceptance-length figures from the DFlash2 model card
  (H200/SGLang) stay clearly labelled as somebody else's hardware.
- **No vLLM number will be quoted anywhere, including as an estimate.**

The diagnosis above is kept so the decision is reversible if the driver is ever updated.

## 3. What this does to the gate targets

Two of the directive's gates are set **below** the measured llama.cpp baseline, which would make
"passing" them a loss. Flagging now rather than at Phase 5.

| Gate | Directive min / stretch | llama.cpp measured | Verdict |
|---|---|---|---|
| G1 — AR decode | 35 / 45 tok/s | **38.41** | **min is a LOSS.** Raise to 45 min / 55 stretch. 55 is 80% bandwidth efficiency on our 12.435 GiB footprint, i.e. matching llama.cpp's efficiency on a smaller footprint. |
| G3 — prefill 8K | 600 / 900 tok/s | **1318** | **badly mis-set.** Raise to 1400 min / 1800 stretch. |
| G4 — spec decode, code | 95 / 130 tok/s | 51.30 (humaneval) | sound; 95 is 1.85x llama.cpp |
| G5 — spec decode, math | 120 / 160 tok/s | 82.94 | sound; 120 is 1.45x llama.cpp |

Note the directive predicted "llama.cpp around 22-30 tok/s AR". It measures **38.41**. The
directive's own warning applies in reverse: the gap was in the estimate, not the benchmark.

### Where our advantage actually comes from
It is not kernel heroics. It is the weight footprint:

| | llama.cpp Q4_K_M | this project (INT4 g128 + INT4 lm_head + INT8 embed) |
|---|---|---|
| weights read per decoded token | ~17.0 GiB | **12.435 GiB** |
| tok/s at 77% bandwidth efficiency (llama.cpp's) | 38.41 | **56.6** |
| tok/s at 70% | 34.9 | 51.5 |

Matching llama.cpp's *efficiency* on our *footprint* is already ~1.47x. That is the honest
shape of the AR win, and it says the highest-leverage work is the `lm_head`/`embed` repack and
the g128-vs-g32 quant choice — not exotic kernels.
