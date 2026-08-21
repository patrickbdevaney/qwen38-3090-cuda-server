# QUANT_CHOICE.md

**Status: OPEN.** The structural analysis is complete and is decisive about what the trade *is*.
The quality half of the bake-off (perplexity, KL vs BF16, HumanEval, GSM8K) has not been run.
No checkpoint is finally chosen. Working assumption for Phases 1–2 is stated in §5.

---

## 1. Classification

Per the directive, with one correction.

**Usable on sm_86 (W4A16 INT4, group-quantized) — downloaded and inspected:**
- `philbert440/Qwen3.8-27B-W4A16-AWQ` — 19.5 GB, compressed-tensors pack-quantized, **g128**
- `cyankiwi/Qwen3.8-27B-AWQ-INT4` — 21.0 GB, compressed-tensors pack-quantized, **g32**

**Usable, not yet downloaded:** `avyukth/...`, `FenomAI/...`, `soyrsoyr/...-GPTQ`.

**Rejected — wrong hardware:**
- `amd/...-Quark-AWQ-MXFP4`, `TelperionAI/...-NVFP4-AWQ-AutoRound` — Ampere has no FP4 path;
  software dequant throws away the only reason to use the format.

**Budget-hostile, bench only for quality reference:** `True2456/...-4.85bpw`, `...-5.0bpw`.

**Reference cross-check, not a build target:** `petr567/Qwen3.8-27B-Q4_K_M-DFlash2-Strix-Halo`.
*Correction to the directive*: that repo's only file is a **1.14 GB drafter GGUF**, not a 27B
model. The llama.cpp baseline is `ggml-org/Qwen3.8-27B-GGUF` Q4_K_M (18.97 GB), which is what
`reports/BASELINES.md` measures.

---

## 2. On-disk layout — both candidates, verified from safetensors headers

**Both use the identical container**, which is good news for the loader: one unpack path.

| | value |
|---|---|
| quant_method | `compressed-tensors` |
| format | `pack-quantized` |
| bits | 4 |
| symmetric | **false** — explicit zero points, asymmetric |
| observer | `mse` |
| strategy | `group` |
| tensors | `weight_packed` I32 (8 int4 per word), `weight_scale` BF16, `weight_zero_point` I32 (8 per word along the **output** dim), `weight_shape` I64[2] |

This is **not** classic AWQ `qweight/qzeros/scales` with GPTQ interleave. The exact bit order
within each I32 word must be pinned against a reference dequant in Phase 1 before any kernel
consumes it — that is Gate G1.

Example (philbert440, `mlp.gate_proj`, true shape `[17408, 5120]`):
`weight_packed [17408, 640]` (5120/8), `weight_scale [17408, 40]` (5120/128 → **group 128**),
`weight_zero_point [2176, 40]` (17408/8 packed along output).

### `lm_head` and `embed_tokens` are BF16 in BOTH
`lm_head.weight` and `model.language_model.embed_tokens.weight` are both
`BF16 [248320, 5120]` = **2.368 GiB each** in both checkpoints. Neither quantizer touched them.
This is the directive's failure-mode #4, confirmed and doubled — 4.736 GiB in two tensors.

The vision tower (0.858 GiB) is in the quantizer's `ignore` list and stays BF16; we skip it.
The MTP head ships separately (philbert440: `model-mtp.safetensors`, 0.791 GiB).

---

## 3. The trade, quantified

| | philbert440 | cyankiwi |
|---|---|---|
| group size | **128** | **32** |
| effective bpw (packed+scales+zeros) | **4.156** | **4.625** |
| quantized params | 24.327 B | 24.296 B |
| body on device | **11.820 GiB** | **13.188 GiB** |
| decode weight traffic/token (+ INT4 lm_head) | **12.435 GiB** | **13.803 GiB** |
| AR ceiling at measured 914.2 GB/s | **68.5 tok/s** | 61.6 tok/s |
| KV left at 128K target | 4.305 GiB | 2.937 GiB |
| max FP8 context | **141,074 tok** | 96,240 tok |
| other | separate MTP file, `recipe.yaml` | leaves layer 0 `linear_attn.out_proj` BF16 |

**cyankiwi's g32 costs 1.368 GiB.** In this build that buys, or rather spends:
- **11% of the AR decode ceiling** (68.5 → 61.6 tok/s), and
- **~45,000 tokens of context** — enough to put the 128K headline claim out of reach.

g32 has 4x the scale/zero-point traffic of g128 and its scales are a real fraction of the read
(1.414 GiB vs 0.380 GiB). So the cost is not just capacity, it is bandwidth on the hot path.

**This is the whole decision.** It is not a file-size preference; it is 11% of the product's
headline number and the 128K claim, against whatever g32 buys in quality. Per the directive it
gets settled on KL divergence and task score.

---

## 4. Bake-off protocol — TO RUN

`tools/quant_bakeoff.py`, once, results committed. Reference is the BF16 checkpoint
(`Qwen/Qwen3.8-27B`, 55.6 GB) run with CPU offload; 60 GiB host RAM makes this feasible but slow.

1. Perplexity on a 512-sample held-out mix (code, prose, math, tool-call transcripts) vs BF16.
2. **Top-1 agreement rate and mean KL divergence vs BF16 logits over 20k tokens** — the primary
   criterion.
3. HumanEval subset (50 problems) and GSM8K subset (100 problems), greedy.
4. On-disk layout (done, §2).

Two extra cells this build needs, not in the directive's list, because they decide 2.94 GiB:

5. **`lm_head` BF16 → INT4 g128**: KL and top-1 agreement against the same checkpoint with the
   BF16 head. Saves 1.753 GiB. This is the single most quality-sensitive tensor in the model and
   it is also the drafter's output head (DFlash2 has none of its own), so a regression here hits
   acceptance length as well as output quality.
6. **`embed_tokens` BF16 → INT8 rowwise**: saves 1.183 GiB. Should be nearly free — it is a row
   gather, not a GEMM input — but it must be measured, not assumed.

Environment is provisioned: `transformers 5.15.1` + `compressed-tensors 0.18.0` in
`qwen38-weights/hfenv`, verified to parse the config as `Qwen3_5Config`. (The user's existing
venv has transformers 4.56.0, which cannot load this architecture at all.)

---

## 5. Working assumption, pending the bake-off

**Phases 1–2 target `philbert440/Qwen3.8-27B-W4A16-AWQ` (g128).** Reasons:

1. It is the configuration the whole VRAM budget and the 128K claim depend on.
2. g128 is the *harder* kernel target — wider groups, fewer scale reads per output, so the
   dequant-fused GEMV has less scale traffic to hide behind. Code written for g128 handles g32
   by changing a constant; the reverse is not true.
3. Switching later is a loader change, not a kernel change.

**This is not the final decision and is not presented as one.** If the bake-off shows g128 costs
materially more than g32 in KL or task score, the honest outcome may be that 128K context and
best-quality-4-bit are not simultaneously available on 24 GB, and that becomes a documented,
user-selectable trade (`--quant g32 --max-context 96000` vs `--quant g128 --max-context 131072`)
rather than a silent choice.
