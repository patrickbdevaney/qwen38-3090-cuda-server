# A GGUF loader in the server: what works, and the three conventions that do not

Status: **loads, runs end to end, and produces wrong logits.** The plumbing is
done and gated; the remaining work is that llama.cpp's GGUF conversion applies
transforms to the GDN tensors that this loader does not yet undo. This report
records exactly which, so the next session starts from a list rather than a
symptom.

```
./build/cuda_server --model <hf_or_awq_dir> --gguf <file.gguf>
```
`--model` still supplies `config.json` and `tokenizer.json`; only the weights
move. A GGUF carries equivalent metadata (`qwen35.*` keys are complete enough to
rebuild the shape) but this server's shape parser and tokenizer are gated
against the HF files, and one source of truth for the shape is worth more than
independence from the directory.

## What is done

**`Linear`.** AWQ ships q|k|v -- and GDN's qkv|z, and gate|up -- pre-concatenated
into one packed tensor. GGUF ships them separately and lets each carry its own
quant type: `blk.3` of UD-Q3_K_XL has `attn_q` as IQ4_NL, `attn_k` as Q4_K and
`attn_v` as Q5_K. They can neither be concatenated at load nor read by one
kernel. `Linear` holds either shape, and the GGUF kernels grew an `ldy` stride so
each part writes its own column range of one `[T, out_f]` buffer. Six call sites
in `run_layer` were all that needed to change.

**Loading.** `src/model/load_gguf.cu` fills the same `Model` the AWQ loader
does. Big projections stay in their block format and go through the fused GEMV;
norms, biases and conv kernels are dequantised once to bf16 at load. `token_embd`
is dequantised to **host** bf16 -- it is a row gather, not a matmul, so it costs
2.42 GiB of host RAM instead of VRAM the KV cache wants more. `output.weight`
stays in blocks and runs through the fused GEMV rather than being re-quantised
to our INT8, which would be a second lossy step on top of the file's own.

**It runs.** 8192 context, greedy, 24-25 tok/s, no crashes, and the footprint is
what the format promises:

| | AWQ INT4 | GGUF UD-Q3_K_XL |
|---|---:|---:|
| body | 11.859 GiB | **10.550 GiB** |
| lm_head | 1.203 GiB (INT8) | **0.814 GiB** (blocks) |
| embed | host | host |
| total weights | 13.06 GiB | **11.36 GiB** |

1.70 GiB less, which at 18 KiB/token of INT4 KV is **99K more tokens of
context** -- the reason for doing this at all.

## What is wrong

`gate_gguf_model` runs the same prompt through both checkpoints and compares the
residual stream layer by layer. Layer 0 (the embedding) tracks to 0.12 relative,
which is just Q3_K against bf16. Layer 1 is already 3.9 and it never recovers;
teacher-forced logits score KL 16.0 and 1.4% top-1 against BF16, i.e. noise.

The same gate compares the small per-layer tensors directly, and that is where
the answer is. Three separate conventions, all of them llama.cpp's:

**1. V-head interleaving.** llama.cpp's own comment in `llama-model.cpp` states
it outright:

> both Qwen 3 Next and Qwen 3.5 support n_v_heads > n_k_heads but the
> broadcasting pattern is different:
>   - Qwen 3 Next: `[k0_v0, k0_v1, k1_v2, k1_v3]`
>   - Qwen 3.5:    `[k0_v0, k1_v1, k0_v2, k1_v3]`

HF blocks the 48 v-heads under the 16 k-heads (`v_head / 3`); the GGUF
interleaves them (`v_head % 16`). Confirmed against the data: `ssm_dt.bias`
element 1 in the GGUF equals element 3 in the AWQ checkpoint, and
`p -> (p % 16) * 3 + (p / 16)` predicts it. This permutes every per-v-head
tensor -- `ssm_dt.bias`, `ssm_a`, and the rows of `ssm_alpha`/`ssm_beta` -- and
segments `ssm_conv1d`, `attn_qkv`, `attn_gate` and `ssm_out` along their
key_dim-sized blocks.

**2. `ssm_a` is not `A_log`.** llama.cpp stores the transformed value. AWQ
`A_log[0] = -3.20312`; GGUF `ssm_a[0] = -0.04053`, and `-exp(-3.20312) =
-0.0406`. The loader must invert it, or the kernel must learn the other
convention.

**3. The norms disagree and it is not yet understood.** AWQ `input_layernorm`
starts `0.04688, 0.26562, 0.45312`; GGUF `attn_norm` starts `1.04688, 0.93750,
0.92578`. Element 0 differs by exactly 1, which suggests a `1 + w` convention,
but element 1 does not fit that at all, so the simple explanation is wrong.
Either these are different tensors under similar names, or a transform is being
applied that the first element coincidentally satisfies. `ssm_norm` matches
bit-exactly, so it is not a blanket difference. **This one needs the actual
conversion script for Qwen3.5**, which is not in the llama.cpp checkout on this
box -- it only carries the `qwen35` constants, so the Unsloth GGUFs were built
with a newer tree.

## Why this is committed while broken

The infrastructure is right and independently useful: `Linear` and the strided
kernels are what any non-AWQ format needs, and they are what an EXL3 backend
would reuse. Every one of the 20 existing gates still passes, so the AWQ path is
untouched. `gate_gguf_model` is deliberately NOT registered in ctest -- it fails,
and a suite that is red by design teaches people to ignore it. It is a
diagnostic to run by hand until the three items above are fixed, at which point
it becomes the gate.

What it must not be is described as working. `--gguf` currently produces fluent-
looking nonsense, which is the most dangerous failure mode a quantisation path
has, and the flag should carry that warning until the residual tracks.
