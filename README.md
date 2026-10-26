# qwen38-3090-cuda-server

A pure-CUDA inference engine and OpenAI-compatible server for **Qwen3.8-27B** on one **RTX 3090** (24 GB, `sm_86`),
with speculative decoding through the **DFlash2** block-diffusion drafter. It is built for one job: serving a local
agentic coding harness (Claude Code-, Codex- or OpenCode-style clients). That job needs long context, prefix reuse
across turns, exact greedy decoding for tool calls, and fast decode. The request path has no Python: it runs
hand-written kernels, a C++ tokenizer and chat template, and an HTTP server in one binary.

## Status (last measured 2026-08-23)

This is a research release from a single maintainer. It serves one sequence at a time, at up to the model's trained
262,144-token context, from either of two weight formats.

| Component | State |
|---|---|
| AWQ INT4 runner ([`philbert440/Qwen3.8-27B-W4A16-AWQ`](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ)) | Primary path; fastest |
| GGUF runner ([`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), UD-Q3_K_XL) | Same accuracy as AWQ, 1.7 GiB lighter, slower decode |
| DFlash2 speculative decode ([`z-lab/Qwen3.8-27B-DFlash2`](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)) | Lossless acceptance rule; adaptive on/off per request |
| Prefix cache, INT4 KV cache, CUDA graphs, vision (`--vision`) | Working |
| Tests | 22 registered gates pass (`ctest`); the live-server gate passes 42/42 on both formats |
| Concurrent batching, multi-GPU, CPU offload | Out of scope |

## Results

Every figure comes from a committed binary, invocation and log on one RTX 3090
([`reports/BENCHMARKS.md`](reports/BENCHMARKS.md)). Projections are labeled as projections.

**Decode through the HTTP endpoint.** INT4 KV, DFlash2 loaded, `enable_thinking: false`, `temperature: 0`, 192
tokens:

| Prompt | AWQ AR | AWQ speculative | Accepted / round of 8 | GGUF AR | GGUF speculative |
|---|---:|---:|---:|---:|---:|
| Count to 60 | 45.9 | **148.8** | 8.00 | 35.5 | **93.7** |
| Reverse a linked list (code) | 45.6 | **142.0** | 7.58 | 35.2 | **88.8** |
| Explain quicksort (prose) | 45.6 | 49.0 | 2.71 | 35.1 | 32.8 |

- **Prose barely gains.** At 2.4–2.7 accepted tokens per round, speculation roughly breaks even.
- **The server switches speculation off when it doesn't pay.** It measures both rates and picks the faster.
  Without that switch the GGUF prose row was 27.9 tok/s, against 30.1 plain.

**Against baselines on the same box, same prompts and harness** ([`reports/BASELINES.md`](reports/BASELINES.md)):

| | Decode tok/s |
|---|---:|
| llama.cpp Q4_K_M, autoregressive | 38.41 |
| llama.cpp Q4_K_M + MTP speculation | 51.30 median (82.94 math, 43.96 prose) |
| This server, AWQ, autoregressive | **45.8** |
| This server, AWQ + DFlash2 | **100.1** mean over three prompt classes; 142–149 on code and structured output |

- **llama.cpp is a strong baseline.** It reaches 77% of measured DRAM bandwidth.
- **The AR advantage is mostly footprint.** The AWQ path reads 12.4 GiB per token against ~17.0 for Q4_K_M, and
  its GEMV runs at 84% of DRAM.
- **llama.cpp cannot run DFlash2 at all.** Its `draft-dflash` architecture is DFlash 1: it creates 58 of the
  drafter's 81 tensors and aborts.
- **vLLM was not measured.** It would have needed a driver upgrade that previously broke suspend on this machine.

**Accuracy against the true BF16 model.** `Qwen/Qwen3.8-27B` was streamed layer by layer and teacher-forced on
15,520 agentic-coding positions ([`reports/QUANT_ACCURACY.md`](reports/QUANT_ACCURACY.md)):

| Weights | KL mean | KL p99 | Top-1 | Resident |
|---|---:|---:|---:|---:|
| AWQ INT4 g128 + INT8 `lm_head` | 0.150 | 4.26 | 98.56% | 13.06 GiB |
| GGUF UD-Q3_K_XL (this server's GGUF runner) | 0.143 | 4.27 | 98.57% | **11.36 GiB** |
| EXL3 3.50 bpw, VRAM-matched (exllamav3) | 0.131 | 3.49 | 98.51% | ~14.5 GiB |

The median KL is 5e-05 but p99 is 4.26. Quantization error concentrates on a few positions, so quant decisions here
are made on the tail, not the mean.

**Long context and multi-turn**

| Metric | Value |
|---|---|
| Context | 262,144 tokens (trained max), with the drafter loaded; peak 21,862 MiB with INT4 KV |
| Allocatable beyond that | ≥ 360,448 tokens, but positions past 262,144 are untested (RoPE extrapolates) |
| Decode at 262K | 24.5 tok/s |
| INT4 KV cache vs FP8 | 3.58 GiB freed; KL 3.7e-04 at 131K tokens; needle retrieved; same decode speed |
| Prefix cache, second turn | **90.5×** faster prefill (6,715 of 6,736 tokens reused); snapshot restore 21 ms |

## Findings

**1. The hybrid layout is what makes 262K fit on 24 GB.**
- Qwen3.8-27B has 48 GatedDeltaNet (linear-attention) layers and 16 full-attention layers. Only the 16 hold a KV
  cache: 18 KiB per token at INT4, against the 4× a dense 27B would need.
- The recurrent state is 144 MiB and constant in context length.
- Repacking the BF16 `lm_head` to INT8 and the embedding table to INT8 reclaims 2.94 GiB. That is the difference
  between ~35K and 128K of context. Both repacks are gated on KL, not assumed free.
- `--embed-host` moves the embedding table to host memory: it is a row gather, one 10 KB row per token over PCIe.

**2. INT4 KV buys memory, not speed.** Decode attention was already ALU- and latency-bound (444 GB/s against a
788 GB/s load-only ceiling). Dequantization fills exactly the bandwidth that halving the cache frees. Keys and values
are configurable independently (`--kv-cache fp8 | v4 | int4`), because key noise moves where attention lands while
value noise averages out.

**3. The GGUF gap is the formats' codebook arithmetic, not the kernel shape.**
- The fused GGUF GEMV went from 355 to 598 GB/s, which took GGUF decode from 23.8 to 35.5 tok/s.
- Q6_K, whose dequantizer is a shift and a subtract, runs at 90% of DRAM in the same kernel.
- IQ4_XS and IQ3_S, two-thirds of the file by bytes, sit at 67–71%, because every value costs a codebook lookup.
- GGUF reads 1.7 GiB less per token than AWQ, so its ceiling (72.6 tok/s) is higher. The remaining gap is unfinished
  kernel work.
- The rejected experiments are recorded in [`reports/CONFIG_MATRIX.md`](reports/CONFIG_MATRIX.md): shared-memory
  activations, accumulator splits, warps-per-block sweeps and L1 bypass.

**4. Serving bugs hid from all 21 kernel gates.** The live-server gate found them.
- **`temperature: 0` was not greedy.** Qwen's default `presence_penalty` of 1.5 was applied anyway. That changed
  which token was the argmax on code, and it silently disabled speculation, because acceptance is argmax equality.
  The standard coding-agent request went from 43 to 146 tok/s once fixed.
- **`chat_template_kwargs` was ignored**, so harnesses could not turn thinking off.
- **Stop sequences were missed, then undone** by a post-loop flush.
- **One token can carry half of a UTF-8 character.** Streaming it terminated the whole process; partial sequences
  are now held back until complete.

**5. Speculation is lossless in its acceptance rule, but not bit-identical.** Longest-prefix argmax equality plus the
target's correction emits what the target would have emitted. Verification at width 8, however, runs different
kernels from single-token decode, and sums in a different order. `gate_spec` therefore requires every divergence to
be an exact bf16 tie: 1 divergence in 4 × 192 tokens, at a logit gap of exactly one ulp. Requests can set
`"speculative": false` for reproducibility across retries.

## Getting started

Requires an RTX 3090 (`sm_86`) with CUDA 12.x and CMake. Weights are not in this repository.

```bash
huggingface-cli download philbert440/Qwen3.8-27B-W4A16-AWQ --local-dir ~/models/qwen38-awq
huggingface-cli download z-lab/Qwen3.8-27B-DFlash2        --local-dir ~/models/qwen38-dflash2

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DQWEN38_MODEL_DIR=$HOME/models/qwen38-awq \
      -DQWEN38_DRAFT_DIR=$HOME/models/qwen38-dflash2 \
      -DQWEN38_BF16_DIR=/path/to/bf16-reference        # needed only by the accuracy gates
cmake --build build -j
ctest --test-dir build

./build/cuda_server --model ~/models/qwen38-awq --draft ~/models/qwen38-dflash2 \
                    --kv-cache int4 --max-context 262144 --embed-host --port 8080
```

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Write a Python function that reverses a linked list."}],
  "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}'
```

**Options**

| Flag | Effect |
|---|---|
| `--gguf FILE` | Run a GGUF checkpoint instead of AWQ |
| `--kv-cache fp8\|v4\|int4` | KV format: FP8 / FP8 keys with INT4 values / INT4 |
| `--embed-host` | Embedding table in host memory (+1.185 GiB of device memory) |
| `--vision`, `--vision-max-patches` | Load the 27-block vision tower (0.858 GiB); images as `data:` URLs or local paths |
| `--prefix-slots`, `--no-prefix-cache` | Prefix-cache snapshots, stored in pinned host memory |
| `--no-graph`, `--prefill-chunk`, `--webui` | Debugging and tuning |

- **API:** OpenAI chat and completions with streaming, tool calls and a separate reasoning field.
- **Context discovery:** `/v1/models` advertises the context window, so a harness can decide when to compact.
- **Images:** remote image URLs are never fetched, and video is rejected.
- **Debugging:** `QWEN_DEBUG_SYNC=1` aborts at the first faulting kernel; `=2` prints a per-stage profile.

## Compatibility

| Platform | State |
|---|---|
| RTX 3090 (GA102, `sm_86`), 24 GB | Tested |
| Other `sm_86` cards with 24 GB (A5000, A10) | Expected to work; not tested |
| Other architectures or smaller cards | Not supported: kernels and VRAM budget target this card |
| Other GGUF quants of Qwen3.8-27B | All 13 GGUF types dequantize bit-exact against ggml; only UD-Q3_K_XL was measured end to end |

## Limitations

- **Single stream.** One sequence plus one background prefill slot; no continuous batching.
- **Two pre-registered gates missed.** Decode at 64K is 82% of the 4K rate against an 85% bar (G2). Prose
  speculation (73–105 tok/s) is below its 120 tok/s bar (G5).
- **Decode attention dominates at long context.** It is 6% of a step at 4K and 39% at 131K. A base-2 softmax was
  measured at +23% for that kernel and rejected, because it changed argmaxes beyond ties.
- **The prefix cache does not snapshot the drafter's state,** so acceptance dips briefly after a cache hit.
- **Accuracy is measured as agreement with BF16 on agentic-coding text,** not as downstream benchmark scores.

## Roadmap

1. **RoPE extension past 262,144** (YaRN-style), with measured quality. Memory already allows 360K.
2. **Finish the GGUF GEMV.** At AWQ's efficiency, GGUF would decode at ~52 tok/s on fewer bytes.
3. **Drafter state in prefix snapshots.**
4. **An `sm_86` EXL3 kernel.** It is the most accurate format at matched VRAM; exllamav3's fast paths need
   `sm_89`+.

## Repository map

| Path | Contents |
|---|---|
| [`src/kernels/`](src/kernels/) | W4A16 GEMV and tensor-core GEMM, GatedDeltaNet, attention (FP8/INT4 KV), sampling |
| [`src/model/`](src/model/), [`src/gguf/`](src/gguf/) | Loaders, layer stack, CUDA graph capture, VRAM budget; GGUF container, 13 dequantizers, fused GEMV |
| [`src/draft/`](src/draft/), [`src/spec/`](src/spec/) | DFlash2 drafter and candidate selector; speculative loop, GDN state rollback, adaptive routing |
| [`src/cache/`](src/cache/) | Prefix cache: recurrent-state snapshots in pinned host memory |
| [`src/vision/`](src/vision/) | 27-block ViT, preprocessing, mrope and splicing |
| [`src/tokenizer/`](src/tokenizer/), [`src/server/`](src/server/) | BPE, NFC, chat template; OpenAI endpoints, SSE, tool calls, reasoning parser |
| [`tests/`](tests/) | One gate executable per component, including the live-server gate |
| [`bench/`](bench/) | Hardware microbenchmark, per-kernel and end-to-end benches, prompt suite |
| [`tools/`](tools/) | One-time Python: reference dumps, checkpoint inspection, BF16 accuracy comparison |
| [`reports/`](reports/) | `PHASE_0`–`PHASE_9`, [`BENCHMARKS`](reports/BENCHMARKS.md), [`BASELINES`](reports/BASELINES.md), [`QUANT_ACCURACY`](reports/QUANT_ACCURACY.md), [`CONFIG_MATRIX`](reports/CONFIG_MATRIX.md), [`GGUF_LOADER`](reports/GGUF_LOADER.md), [`EXL3`](reports/EXL3.md), [`ROADMAP`](reports/ROADMAP.md), and logs |

## Contributing and contact

Issues are welcome for reproducibility problems and wrong numbers. Pull requests are not being accepted yet.

## License

- **Code in this repository:** MIT ([`LICENSE`](LICENSE)).
- **Third-party code** in [`third_party/`](third_party/) keeps its own license: `httplib.h` (MIT), `json.hpp` (MIT),
  `stb_image.h` (public domain or MIT).
- **Weights** are not distributed here. The target, quantized checkpoints and drafter carry their publishers'
  licenses.

## Citation

```bibtex
@misc{devaney2026qwen38_3090,
  title  = {qwen38-3090-cuda-server: Qwen3.8-27B with DFlash2 speculative decoding on one RTX 3090},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/qwen38-3090-cuda-server}
}
```

Builds on Qwen3.8 (Qwen team), DFlash2 (z-lab), AWQ, and the GGUF formats from llama.cpp/ggml.
