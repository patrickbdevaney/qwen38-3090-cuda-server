# QUANT_CHOICE.md

**Status: DECIDED — `philbert440/Qwen3.8-27B-W4A16-AWQ` (INT4, group 128).**

The structural analysis below is complete and measured. The BF16-reference half of the bake-off
(perplexity / KL vs BF16, HumanEval, GSM8K) was **deliberately not run**: it requires the 55.6 GB
BF16 checkpoint and an hour or two of CPU-offloaded evaluation, and the operator's call was that
this is not worth the time when the structural trade is already decisive. §5 states the decision
and, honestly, what it rests on and what remains unverified.

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

## 5. Decision: philbert440 / g128

### What the choice rests on

1. **The two checkpoints differ in exactly one parameter.** Both are llm-compressor
   `compressed-tensors` / `pack-quantized`, 4-bit, **asymmetric**, **MSE observer**, `strategy:
   group`. Same tool, same method, same calibration philosophy. Only `group_size` differs,
   128 vs 32. That makes the quality ordering knowable a priori — finer groups cannot be worse —
   so the open question was never *which is more accurate*, only *by how much*.
2. **The cost of g32 is measured and large.** 1.368 GiB more body weight, which is
   **11% of the AR decode ceiling** (68.5 → 61.6 tok/s) and **~45,000 tokens of context**
   (141,074 → 96,240). Its scale tensors alone are 1.414 GiB against g128's 0.380 GiB, and those
   scales are read on the hot path, so the penalty is bandwidth as well as capacity.
3. **g32 forecloses the project's headline result.** 128K FP8 KV needs 4.000 GiB; g32 leaves
   2.937 GiB. The whole reason to build this on a 3090 is that the 3:1 GDN layout makes 128K fit.
   A checkpoint choice that takes that away is not a neutral quality upgrade.
4. **The published behaviour of group size at this scale is not close to the cost.** For 4-bit
   weight-only quantization of models in the 20–30 B range, g128 → g32 is a small accuracy
   move; the cost here is 11% of throughput and a third of the context. The burden of proof
   sits with g32, and nothing in the structural data discharges it.
5. **g128 is the harder kernel target**, so writing for it is the conservative engineering
   choice: wider groups mean fewer scale reads per output and less traffic to hide dequant
   behind. Code written for g128 handles g32 by changing a constant; the reverse is not true.

Secondary: philbert440 ships the MTP head as a separate `model-mtp.safetensors` (0.791 GiB),
which is convenient for the Phase 9 MTP-vs-DFlash2 ablation, and a `recipe.yaml` recording the
quantization run.

### What this does NOT rest on, stated plainly

**No KL divergence or task score was measured against BF16.** The directive asked for the
decision to be made on KL and task score; it is instead being made on the structural trade plus
a prior about group size, because the operator judged the BF16 evaluation not worth its cost.
That is a real weakening of the evidence and it is recorded here rather than papered over.

Note also that Phase 5's gate — token-exact match against HF transformers on the *same quantized
weights* — validates our **implementation**, not this **checkpoint choice**. Those are different
claims and must not be conflated in `BENCHMARKS.md`.

One detail worth carrying forward: **cyankiwi leaves layer 0's `linear_attn.out_proj` in BF16**
while quantizing the other 47. Its author evidently judged that tensor sensitive. Our own
`lm_head` INT4 and `embed_tokens` INT8 repacks should be gated the same way — measured, with a
per-tensor escape hatch — rather than applied blindly.

### The escape hatch

The loader reads group size from `quantization_config` and never hardcodes it, so both
checkpoints load through one path. If Phase 5 or Phase 9 turns up a quality problem traceable to
g128, switching is a `--model` flag, not a rewrite, and the honest consequence is documented:

```
--model philbert440-g128  --max-context 131072   (default)
--model cyankiwi-g32      --max-context  96000   (higher-fidelity 4-bit, less context, ~11% slower)
```

### Deferred, cheap, worth doing in Phase 9

Once the server exists, comparing the two checkpoints costs almost nothing and needs **no BF16
model**: run both through our own runtime on the same prompts and report top-1 agreement between
them plus HumanEval/GSM8K scores for each. That bounds the disagreement without a 55.6 GB
download. It cannot say which is closer to BF16 — only how far apart they are — but if they
agree at >99.5% top-1 the question is settled for practical purposes.
