# PHASE 0 — Recon

Status: **Gate 0 PASSED.** No kernels written. Baselines partially measured (§6).

Every number below is either measured on this box with a committed binary and a committed log,
or is derived arithmetically from a measured number. Where the directive supplied an estimate
and the measurement disagrees, the measurement wins and the delta is called out.

---

## 1. The box

`bench/microbench.cu` → `reports/logs/microbench_sm86.log`.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3090, **sm_86**, 82 SMs |
| VRAM | 23.57 GiB total, **22.64 GiB free** (0.93 GiB held by Xorg + gnome-shell + rustdesk + firefox) |
| SM clock | 1710 MHz reported max, **1965 MHz measured peak boost** |
| Memory | 384-bit GDDR6X @ 9751 MHz → 936.1 GB/s theoretical |
| L2 | 6.00 MB |
| Shared memory | **99 KB/block opt-in**, 100 KB/SM |
| Registers | 65536/SM |
| Host | Ryzen 9 5900X, 24 threads, 60 GiB RAM, 585 GB free on NVMe |
| Toolchain | CUDA 12.1 (driver 570.133.07 / CUDA 12.8 capable), GCC 11.5, CMake 3.30.3 |

**A display is attached.** The directive told us to assume 22.3 GB in that case; the measured
free figure is 22.64 **GiB** (= 24.3 GB), i.e. we have *more* headroom than assumed, because the
directive's table mixed GB and GiB. All budgets below are in GiB against the measured 22.64.

### Measured microbenchmarks

| Quantity | Measured | Directive assumed | Delta |
|---|---|---|---|
| DRAM streaming read (2 GiB) | **914.2 GB/s** (2.35 ms, p95 2.36) | "assume 800–840" | **+9 to +14%** |
| DRAM copy read+write | 839.7 GB/s effective | — | — |
| L2-resident read (4 MB, 200 in-kernel reps) | 8603 GB/s | — | 9.4x DRAM |
| BF16 `mma.sync.m16n8k16` → f32 | **81.6 TFLOPS** | — | — |
| `cp.async.cg` vs `LDG+STS`, 16 B/thread | 904.5 vs 907.3 GB/s (**1.00x**) | — | see below |

Two of these need honest annotation:

**The 81.6 TFLOPS is real, not a benchmark bug.** It exceeds the 71.8 TFLOPS datasheet figure
computed at the reported 1710 MHz max clock. Sampling `clocks.sm` during the run gives a peak of
**1965 MHz**, at which 82 SM x 256 FMA/clk x 2 = 82.5 TFLOPS — so the measurement is 99% of peak
at the clock the card actually ran. This is an *issue-rate* ceiling on register-resident
fragments; no real kernel will approach it.

**The cp.async result measures nothing useful and should not be cited.** At 16 B/thread with no
compute, both paths saturate DRAM, so they tie by construction. cp.async's value on Ampere is
register pressure and multi-stage pipelining, neither of which this microbenchmark exercises.
Re-measure inside the real GEMM in Phase 2, not here.

**Arithmetic-intensity balance point: 81.6e12 / 914.2e9 = 89.3 FLOP/byte** (directive said ~76).

---

## 2. GATE 0 — architecture verification

`tools/inspect_model.py` derives every shape from `config.json` + safetensors headers and asserts
it against the directive's table. Logs: `reports/logs/gate0_{philbert440,cyankiwi}.log`.

**Result: every single row matches. Zero mismatches on both candidate checkpoints.**

Confirmed exactly as the directive states: hidden 5120, vocab 248320, 64 layers, 48 GDN + 16 full
attention in a strict `16 x (L,L,L,A)` pattern with attention at indices 3,7,…,63; GDN 48 V-heads
/ 16 QK-heads / head_dim 128; attention 24 Q / 4 KV / head_dim 256; `partial_rotary_factor 0.25`
→ 64 rotary dims and **192 unrotated**; `mrope_section [11,11,10]` (sums to 32 = rotary_dims/2,
tiles exactly), `mrope_interleaved: true`; FFN 17408; `rope_theta 1e7`; context 262144.

### The four things the directive explicitly told us to confirm

1. **`tie_word_embeddings: false`.** `lm_head` is a separate `[248320, 5120]` tensor, and in
   **both** AWQ checkpoints it ships **BF16 and unquantized** — 2.368 GiB. So does
   `embed_tokens`, another 2.368 GiB. This is directive failure-mode #4, confirmed and doubled.
   Our repack reclaims **2.94 GiB** of it (§4).
2. **GDN state and conv width.** `linear_conv_kernel_dim = 4` → conv state is 3 timesteps.
   The conv runs over the concatenated q,k,v = `16*128*2 + 48*128 = 10240` channels
   (`conv1d.weight` is `[10240, 1, 4]`, confirming the fusion). Recurrent state is
   `48 layers x 48 v-heads x 128 x 128 = 37,748,736` elements = **144.0 MiB fp32 / 72.0 MiB bf16**,
   constant in context length. Conv state adds 5.6 MiB. `mamba_ssm_dtype: float32`.
3. **QK-norm and gating.** `q_norm`/`k_norm` present on all 16 attention layers, `[256]` each
   (per-head-dim RMSNorm). `attn_output_gate: true`, `output_gate_type: swish`. GDN has
   `in_proj_z` (output gate), `A_log`, `dt_bias`, and a `[128]` per-head RMSNorm.
4. **`rope_parameters`** as quoted above. **Note the config ships no YaRN block** — `rope_type`
   is `"default"`. YaRN for >262K is a Phase 6 addition we configure, not something we read.

### Things the directive did not mention that we found

- **GDN input projections are split, not fused**: `in_proj_qkv [10240,5120]`,
  `in_proj_z [6144,5120]`, `in_proj_a [48,5120]`, `in_proj_b [48,5120]`. `a` and `b` are
  per-value-head scalars and ship **BF16 unquantized** in both checkpoints.
- **The MTP head is a full attention layer** (`mtp.layers.0.self_attn.*` + mlp + `fc` +
  `pre_fc_norm_embedding` + `pre_fc_norm_hidden`), 0.791 GiB. It is a real zero-extra-download
  drafter option for the Phase 9 ablation.
- **The vision tower is cleanly namespaced** under `model.visual.*` (27 blocks, 0.858 GiB, BF16,
  in the quantizer's `ignore` list). The loader skips it on a prefix test.
- `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5`. The config was
  written by **transformers 5.10.1 / 5.8.0.dev0**. The user's existing venv has **4.56.0**, which
  cannot load this model at all — a fresh env with transformers 5.15.1 is being provisioned for
  the P1 reference oracle.

---

## 3. Checkpoint classification and the bake-off

Downloaded and inspected: `philbert440/Qwen3.8-27B-W4A16-AWQ` (19.5 GB),
`cyankiwi/Qwen3.8-27B-AWQ-INT4` (21.0 GB), `z-lab/Qwen3.8-27B-DFlash2` (3.85 GB).

**Both AWQ candidates use the identical container format** — llm-compressor
`compressed-tensors` / `pack-quantized`, 4-bit **asymmetric** (explicit zero points), MSE
observer, `weight_packed` I32 with 8 int4 per word, `weight_scale` BF16, `weight_zero_point` I32
packed 8-per-word along the output dim. **One unpack path serves both.** This is *not* classic
AWQ `qweight/qzeros/scales` GPTQ ordering; the exact bit order must be pinned in Phase 1.

They differ in the one way that matters:

| | philbert440 | cyankiwi |
|---|---|---|
| group size | **128** | **32** |
| effective bpw | **4.156** | **4.625** |
| quantized params | 24.327 B | 24.296 B |
| body on device (packed+scales+zeros+norms) | **11.820 GiB** | **13.188 GiB** |
| notable | separate `model-mtp.safetensors`, `recipe.yaml` | leaves layer 0 `linear_attn.out_proj` in BF16 |

**cyankiwi's g32 costs 1.368 GiB more.** Two consequences, both large:

- At 32 KiB/token FP8 KV, 1.368 GiB is **~45,000 tokens of context**.
- Decode weight traffic goes 12.435 → 13.803 GiB/token, so the AR ceiling at measured bandwidth
  drops **68.5 → 61.6 tok/s, an 11% haircut on the headline number**.

So the quant choice is not a rounding decision: it trades ~11% decode speed and ~45K context
against whatever g32 buys in quality. Per the directive that call is made on KL divergence and
task score, not file size. **Decided: philbert440 / g128**, on the structural trade plus a prior
about group size rather than a measured KL — the BF16 evaluation was judged not worth its cost.
`reports/QUANT_CHOICE.md` §5 records both the reasoning and the fact that the KL evidence is
absent. The loader reads group size from `quantization_config` and never hardcodes it, so
switching is a `--model` flag.

**Rejected as directed, one line each:**
- `amd/...-Quark-AWQ-MXFP4`, `TelperionAI/...-NVFP4-AWQ-AutoRound` — Ampere has no FP4 path;
  software dequant discards the only reason to use them.
- `True2456/...-4.85bpw`, `...-5.0bpw` — budget-hostile; bench only to quantify quality given up.
- `petr567/...-GGUF` — reference cross-check only. Note its file is **1.14 GB, the drafter
  alone**, not a 27B. The real llama.cpp baseline is `ggml-org/Qwen3.8-27B-GGUF` Q4_K_M (18.97 GB).

---

## 4. VRAM budget — measured, not projected

`tools/vram_budget.py` → `reports/logs/vram_budget_philbert440.log`. This is the reference
implementation of the startup check `src/main.cpp` must enforce.

| Component | Size | vs directive |
|---|---|---|
| LM body INT4 g128 (+scales/zeros/norms) | **11.820 GiB** | projected ~13.5 GB; **better** |
| `lm_head` INT4 g128 (repacked from BF16) | 0.615 GiB | as projected |
| `embed_tokens` INT8 rowwise (repacked) | 1.185 GiB | as projected |
| DFlash2 drafter [INT8] | 1.792 GiB | as projected |
| GDN recurrent + conv state (fp32) | 0.146 GiB | as projected |
| GDN rollback buffers (2x) | 0.292 GiB | as projected |
| GDN prefix snapshots (8 x bf16) | 0.584 GiB | as projected |
| Activations + prefill workspace | 1.400 GiB | as projected |
| **FIXED TOTAL** | **17.835 GiB** | |
| arena (22.640 free − 0.500 safety) | 22.140 GiB | |
| **REMAINING FOR KV** | **4.305 GiB** | projected 3.5–4.5 GB |

**KV: 16 attn layers x 4 KV heads x 256 dim x 2 = 32,768 elements/token → 32 KiB/token at FP8.**

| ctx | BF16 | FP8 |
|---|---|---|
| 32K | 2.00 GiB | 1.00 GiB |
| 128K | 8.00 GiB | **4.00 GiB** |
| 262K | 16.00 GiB | 8.20 GiB |

**128K FP8 KV fits, with 0.305 GiB to spare — with a display attached.** Max context at FP8 is
**141,074 tokens**; at BF16 only 70,537. The directive's headline claim survives contact with the
real checkpoint. Dropping the drafter to INT4 would buy roughly another 28K tokens.

The repack is what makes it fit: `lm_head` BF16→INT4 saves 1.753 GiB and `embed_tokens`
BF16→INT8 saves 1.183 GiB, **2.936 GiB total — the difference between 128K and ~35K of context.**
Both must be gated on KL divergence in Phase 2.

### Revised roofline
Weight traffic per decoded token = body 11.820 + `lm_head` 0.615 = **12.435 GiB**.

- at the measured 914.2 GB/s: **14.60 ms/token → 68.5 tok/s AR ceiling** (directive said ~56)
- at a realistic 80% of achieved bandwidth: 18.26 ms → **54.8 tok/s**

So gate G1 (35 min / 45 stretch) sits at 51% / 66% of the ceiling. That is a reasonable target;
the published bandwidth efficiencies laguna collected put vLLM/SGLang near 50% and the best
hand-written stacks at 78–82%.

### Speculation cost, re-derived on real numbers
Verifying 8 tokens: `8 x 2 x 25.6e9 = 409 GFLOP` against the same 13.35 GB read →
**30.6 FLOP/byte**, well under the measured 89.3 balance point. The block stays memory-bound, and
in bandwidth terms would stay so out to a block of ~23.

**But the directive's "verification is nearly free" is weaker here than on an H200, for a reason
worth stating now.** 409 GFLOP at the 81.6 TFLOPS issue ceiling is 5.0 ms against 14.6 ms of
memory time — 34%, hideable only if the GEMM actually overlaps well. At a realistic 50% MMA
efficiency it is ~10 ms against 14.6 ms, which is not free. Worse, **at block size 8 the verify
GEMM has M=8, which wastes half of `mma.sync.m16n8k16`'s M dimension.** Phase 2 should either pad
M to 16 or use a multi-accumulator GEMV. This is the first place I expect the directive's
"0.63 is the floor" claim to be tested.

---

## 5. The open questions, ranked

**#1 — the riskiest: drafter precision.** The directive budgets the DFlash2 drafter at INT8
(1.792 GiB) and calls precision "acceptance rate only, never correctness". That is true of
*correctness*. But gemma-cuda-hybrid tested exactly this and recorded that draft→FP8 and
draft→FP4 both collapsed acceptance from 13.33 to 11.14, concluding "the bf16 draft IS the moat."
If that transfers, INT8 drafting costs ~15% of τ, and τ is the throughput multiplier. BF16 costs
1.792 GiB more, which is ~59K tokens of context. **This is a direct context-vs-throughput trade
with no obviously right answer, and it must be settled by measurement in Phase 7, not assumed
now.** It is the single riskiest open question in the build.

**#2 — g128 quality is assumed, not measured.** 11% of the decode ceiling and 45K of context ride
on g128 over g32, and the decision was taken without a KL measurement against BF16 (see
`QUANT_CHOICE.md` §5). Both checkpoints come from the same tool with the same settings and differ
only in group size, so g32 cannot be *worse* — the unquantified part is how much better. The cheap
Phase 9 check (agreement between the two checkpoints through our own runtime, no BF16 needed)
bounds it without a 55.6 GB download.

**#3 — head_dim 256 attention tiling.** Measured: `Br=64,Bc=64` Q+K+V in BF16 is **96 KB against
a 99 KB ceiling** — it fits, but with 3 KB left and nothing for softmax statistics. `Br=64,Bc=32`
is 64 KB and comfortable; keeping K,V in FP8 in shared memory also gives 64 KB at
`Br=64,Bc=64`. All three of the directive's options are live; option (a) is tighter than it
looks. Benchmark all three in Phase 4.

**#4 — the M-dimension waste in verification** described in §4.

---

## 6. Baselines — IN PROGRESS, not yet measured

Per directive §5.3 a win is not a win without both baselines on this box. Neither laguna nor
deepseek ever actually ran llama.cpp or vLLM; both closed their head-to-head gates against
published vendor numbers. **This is therefore new work.**

State at the time of writing:
- **llama.cpp**: **measured.** 38.41 tok/s AR, 51.30 tok/s with MTP speculation, 1318 tok/s
  prefill at 8K. It **cannot** run DFlash2 — its DFLASH arch is DFlash 1. Full numbers and the
  tensor-level diagnosis in `reports/BASELINES.md`.
- **GGUFs**: downloaded — Q4_K_M (18.97 GB), `mtp-...-Q8_0` (3.16 GB), DFlash2 BF16 and Q8_0.
- **vLLM**: **dropped from scope by operator decision.** It is blocked on the driver (every
  release supporting `Qwen3_5` pins torch >= 2.11 → CUDA 13 wheels → driver >= 580; this box has
  570.133.07). Upgrading risks a known suspend/resume regression on this desktop, and a
  from-source CUDA 12.8 build was judged not worth the time for a number that is only a
  comparison point. **llama.cpp is the measured competitor.** `reports/BASELINES.md` §2 keeps
  the full diagnosis so the decision can be revisited if the driver situation changes.
- **transformers 5.15.1 + compressed-tensors 0.18.0**: installed and verified — parses the
  config as `Qwen3_5Config`. Ready for the P1 reference oracle and the quant bake-off.

`reports/BASELINES.md` has the llama.cpp numbers and the protocol fixed (3 warmup, 10 measured, median + p95,
identical prompts and sampling params) and the numbers left empty. **No baseline number will be
quoted anywhere until it comes from a committed log.**

---

## 7. Plan changes

1. **Adopt laguna's gate ladder** rather than the directive's phase list verbatim; they are
   compatible, laguna's is finer. Three amendments, all argued from prior-repo history in
   `PRIOR_ART.md` §4: add gate **B1c** (batched-forward ≡ sequential-decode, bit-exact, on
   uniform-random token ids), split the kernel gate into **G3-GDN / G3-Full**, and add a gate
   asserting the 3:1 layer interleaving (a wrong layer map passes every per-kernel gate).
2. **Do not build tree verification.** The caddtree repo implemented CaDDTree on a GDN hybrid,
   verified it correct, optimized it 3.6x, and concluded it **loses to linear-chain DFlash**.
   Linear chain only. Saves a phase.
3. **GDN rollback is design (A), and the reference validates it** — `model_mlx.py:539-653` does
   capture-seed-and-replay, exactly the directive's option (A), with two refinements we adopt:
   the conv state is *sliced* from the captured window rather than replayed, and only the seed
   state needs a per-layer buffer. Also implement the correctness assertion caddtree *designed
   but never built*: recompute each layer's state from the seed over only the accepted tokens
   and `allclose` against the promoted state.
4. **Cap the drafter's context at 2047 positions.** All 5 draft layers are sliding-window 2048,
   so the reference caps target-hidden context at `sliding_window-1`. This makes draft KV and the
   context projection **constant in sequence length** — decisive at 128K.
5. **Add three server requirements the directive omits**, from the `local-agent-bootstrap`
   acceptance contract: `GET /health`, a llama.cpp-shaped `timings` object on non-streaming chat
   completions, and a configurable model id matching `/v1/models[0].id`. Missing any one silently
   breaks the Phase 8 harness.
6. **`tau` is not part of DFlash2.** The official repo has no threshold of any kind. The knob is
   a third-party lossy addition. Ship the exact rule; if `--accept-eps` is ever added it is
   off by default, T>0 only, and loudly disables blocking test 1.

## 8. Deliverables

| File | State |
|---|---|
| `bench/microbench.cu` + `reports/logs/microbench_sm86.log` | done |
| `tools/inspect_model.py` + `reports/logs/gate0_*.log` | done, **Gate 0 PASSED** |
| `tools/vram_budget.py` + `reports/logs/vram_budget_philbert440.log` | done |
| `reports/PRIOR_ART.md` | done |
| `reports/PHASE_0.md` | this file |
| `reports/QUANT_CHOICE.md` | **decided** — philbert440 g128; see that file for what the decision rests on |
| `bench/bench_openai.py` + `bench/prompt_suite.json` | done |
| `reports/BASELINES.md` | llama.cpp **measured**; vLLM **blocked on driver** |

---

## The single riskiest open question

**Does the DFlash2 drafter survive INT8 quantization with its acceptance length intact?**

Everything about this build's headline number depends on it. The directive assumes drafter
precision is free because it cannot affect correctness. Correctness, yes — but the prior art on
this exact class of drafter says quantizing it cost ~15% of acceptance, and acceptance is the
multiplier on the whole speculative gain. BF16 costs 1.792 GiB, which is 59K tokens of context.
If both matter we may be forced to choose between "128K context" and "3x decode", which is
precisely the trade this project exists to avoid. It cannot be resolved before Phase 7, and it
should be measured early in Phase 7 rather than at the end.
