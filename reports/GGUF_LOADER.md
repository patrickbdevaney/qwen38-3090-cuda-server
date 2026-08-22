# A GGUF loader in the server

Status: **correct, gated, and end to end.** `--gguf <file>` runs the same server
on a GGUF checkpoint instead of the AWQ one, at the same accuracy against the
true BF16 weights, with 1.70 GiB less resident.

```
./build/cuda_server --model <hf_or_awq_dir> --gguf <file.gguf>
```
`--model` still supplies `config.json` and `tokenizer.json`; only the weights
move. A GGUF carries equivalent metadata (`qwen35.*` is complete enough to
rebuild the shape) but this server's shape parser and tokenizer are gated
against the HF files, and one source of truth for the shape is worth more than
independence from the directory.

## Why bother

The KV cache is what an agent runs out of, and INT4 KV costs 18 KiB/token. The
whole argument for a second weight format is the headroom it buys:

| | AWQ INT4 | GGUF UD-Q3_K_XL |
|---|---:|---:|
| body | 11.859 GiB | **10.550 GiB** |
| lm_head | 1.203 GiB (INT8) | **0.814 GiB** (blocks) |
| total weights | 13.06 GiB | **11.36 GiB** |

1.70 GiB, which at 18 KiB/token is **99K more tokens of context**. That is only
worth taking if it is free in accuracy, and measured against the original BF16
checkpoint it is:

| | KL mean | KL p99 | top-1 | top-5 |
|---|---:|---:|---:|---:|
| AWQ INT4 g128 | 1.495e-01 | 4.263 | 98.56% | 99.40% |
| **GGUF UD-Q3_K_XL** | **1.432e-01** | 4.274 | **98.57%** | 99.39% |
| EXL3 3.00bpw | 1.672e-01 | 5.216 | 97.74% | 99.09% |
| EXL3 3.50bpw | 1.308e-01 | 3.493 | 98.51% | 99.46% |

15520 teacher-forced positions, same prompts, same reference. AWQ and Q3_K_XL
are the same point on the accuracy axis and Q3_K_XL is 13% smaller. This
**reverses** the conclusion recorded in `reports/CONFIG_MATRIX.md`, which was
reached before the loader existed and could only compare kernel efficiency.

## What the conversion does to the weights, and how it was found out

llama.cpp's converter rewrites tensors on the way in. None of the rewrites
crash: a checkpoint with any of them un-done loads, runs at full speed, and
emits fluent nonsense. The first version of this loader scored KL 16.0 and 1.4%
top-1 and looked, from the outside, like a working server.

The first attempt at diagnosis diffed GGUF tensors against the **AWQ**
checkpoint, and that oracle is wrong twice over. AWQ's activation-aware scaling
folds a per-input-channel scale `s` out of every quantised linear and into the
norm that feeds it, so `input_layernorm` there is legitimately `(1 + w) / s`
while `q_norm`, which precedes nothing quantised, is plain `w`. On top of that
the AWQ tensors carry INT4 error of their own. Two of the three conventions
below were mis-read from that comparison; the norm one was called
"not yet understood" and blamed on a missing conversion script.

`tools/cmp_gguf_conventions.py` compares against the **original BF16 weights**
instead, and every convention below is exact there -- `rel 0.000e+00`, not
"close":

**1. Norms carry a +1.** `conversion/qwen.py` adds 1 to every `*norm.weight`
except `linear_attn.norm.weight`, because HF's RMSNorm for this model computes
`(1 + w) * x` and ggml's computes `w * x`. Our `k_rmsnorm` follows HF and
applies the +1 itself, so the loader subtracts it back off `attn_norm`,
`post_attention_norm`, `attn_q_norm`, `attn_k_norm` and `output_norm`.
`ssm_norm` is the exception at both ends: ggml does not shift it and our
`k_norm_gate` uses it raw, so it passes through untouched. That asymmetry is
why the tensor looked bit-identical while every other norm looked broken.

**2. `ssm_a` is `-exp(A_log)`, not `A_log`.** `gdn_gates` wants `A_log`, so the
loader takes `log(-x)`. Exact, not approximate: `bf16(log(-x))` recovers the
original bf16 `A_log` bit for bit.

**3. The v heads are re-ordered.** HF groups the 48 value heads under the 16 key
heads (`k = v / 3`); llama.cpp tiles them so `ggml_repeat` can broadcast
(`k = v % 16`), and permutes every per-v-head tensor to match -- the v rows of
`attn_qkv`, all of `attn_gate`, `ssm_alpha`, `ssm_beta`, the v channels of
`ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, and **the columns of `ssm_out`**.

That last one decides the design. A row permutation of a quantised tensor is a
byte-block gather and could be done at load; a **column** permutation cannot be
done in block format at all without dequantising 31M parameters per layer. So
the loader undoes none of it and the pairing is carried as a flag instead:
`GdnDims::v_tiled` turns `hk = h / hv_ratio` into `hk = h % num_k_heads` at the
one place in `k_scan` that pairs a value head with a key head. One modulo per
block, no load-time work, and the two checkpoints stay bit-comparable in the
residual stream even though their internal head order differs.

## Prefill needs a different kernel from decode

The fused GEMV reads quantised blocks directly and tops out at 8 rows. That is
the right shape for decode and exactly the wrong shape for prefill: a
4096-token chunk streamed every weight 512 times, which measured as **42 tok/s
of prefill** -- roughly an hour to fill 200K context.

Above 32 rows `linear_gguf` now dequantises the projection once into a scratch
buffer and hands it to cuBLAS. Traffic goes from `ceil(T/8)` passes over the
tensor to three (read blocks, write bf16, read bf16), and the scratch is sized
at load from the largest part actually in the file -- 178 MiB for this
checkpoint's 5120x17408 FFN tensors. Teacher-forcing 15520 positions went from
about 9 minutes to **2m07s**, with the KL unchanged at 1.432e-01 against
1.431e-01 for the block path, so the two agree numerically as well as in
principle.

The decode-side counterpart: the GEMV now instantiates every `M` from 1 to 8
rather than only the powers of two. Speculation asks for exactly `block_size-1
= 7` rows through the lm_head every round, and 7 used to split into 4+2+1 --
three passes over a 0.814 GiB head, about 2.4 GiB of reads per round instead of
0.8.

## What is gated

* `gguf_model` (ctest) -- runs the same prompt through the AWQ and GGUF
  checkpoints and compares the residual stream layer by layer. Layer 0 is 0.12
  relative (Q3_K against bf16) and the last layer is 0.138, tracking the whole
  way; the wiring bugs it was built to catch showed as 3.86 at layer 1. It also
  compares the small tensors AWQ leaves alone -- conv1d, A_log, dt_bias,
  ssm_norm -- un-tiling the GGUF side first, and those are bit-exact. It does
  NOT compare the norms or alpha/beta, because AWQ folds its scales into them
  and no agreement is possible; that comparison belongs to
  `cmp_gguf_conventions.py` and its BF16 oracle.
* `tools/cmp_gguf_conventions.py` -- every convention above, against BF16.
* `gguf_gemv` and `gguf_dequant` (ctest) -- all 14 block types bit-exact
  against ggml's own dequantiser.
* `tools/quantcmp_*` -- the KL table above.

## What is still open

* **Decode speed.** The composition-weighted GEMV rate is 355 GB/s against
  llama.cpp's ~585 effective, so decode is slower than AWQ's despite reading
  fewer bytes. The two i-quant types that dominate this file -- IQ4_XS at 405
  and IQ3_S at 293 -- still take their codebook lookups from global memory
  rather than shared. See `reports/CONFIG_MATRIX.md`.
* **Q2_K, Q3_K and Q6_K still use byte loads**, because their block sizes (84,
  110, 210) are not multiples of 8 and a 64-bit load would be misaligned. They
  need a padding repack at load time.
* **`nextn`/MTP.** GGUF files for this model carry a `blk.<n_layer>` holding the
  MTP head. This server drafts with DFlash2, so those tensors are skipped.
