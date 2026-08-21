# PHASE 8 — The server

Status: OpenAI-compatible HTTP server, SSE streaming, tool calls, reasoning
effort modes, Prometheus metrics, a single-file web UI, a terminal client, and
speculative decoding wired into the request path.

```
./build/cuda_server --model /path/to/Qwen3.8-27B-W4A16-AWQ \
                    --draft /path/to/Qwen3.8-27B-DFlash2 --port 8090
```

## Endpoints

| | |
|---|---|
| `POST /v1/chat/completions` | streaming and non-streaming, tools, `reasoning_effort` |
| `POST /v1/completions` | text completion |
| `GET /v1/models` | reports `--alias` |
| `GET /health` | llama.cpp-shaped; body ignored, status is the signal |
| `GET /metrics` | Prometheus |
| `GET /` | single-file web UI |

Responses carry a llama.cpp-shaped `timings` object. That is not decoration:
`local-agent-bootstrap`'s `agent bench` hard-exits without it, taking
`agent code` and `agent status` with it.

## Four things the template made me get wrong first

**The stop token is a list.** `config.json` carries a scalar `eos_token_id` of
248044. `generation_config.json` carries `[248046, 248044]`. Reading the scalar
leaks `<|im_end|>` into the end of every single response.

**The model starts inside the reasoning block.** The generation prompt ends with
`<think>\n`, so the model never emits an opening `<think>` tag. A splitter that
waits for one puts the entire chain of thought into `content` and then leaks a
bare `</think>`. `ReasoningSplitter(starts_in_think)` is constructed knowing
which side of the tag it starts on.

**Reasoning effort is a system-prompt injection, not a sampling knob.** The
template turns `reasoning_effort` into one of three sentences prepended to the
system message, defaults to `xhigh`, and raises on anything outside
`{xhigh, medium, low}`. The server reproduces that rather than inventing its own
mapping.

**OpenAI's `tool_call.arguments` is a JSON string; the template does
`arguments|items`.** The server parses the string back into an object before
rendering, or the template iterates over characters.

## Speculation in the request path

The generation loop is built around a queue of committed-but-not-yet-emitted
tokens. A plain request pushes one token per iteration; a speculative round
pushes up to `block_size` at once. Everything downstream — streaming deltas, the
reasoning splitter, stop strings, abort — sees the identical sequence either way.

Two restrictions, both load-bearing:

* **Greedy only.** The acceptance rule is argmax equality; it reproduces greedy
  decoding and nothing else. A sampled request falls back to plain decode rather
  than silently changing its distribution.
* **No penalties.** Presence, frequency and repetition penalties change *which
  token is the argmax*, and the speculative path takes a raw argmax over the
  target's logits. A request with any penalty active is not speculated, or it
  would produce different text from the same request with speculation off.

When a request did run speculatively, `timings` gains `draft_n`,
`draft_rounds` and `draft_accepted_per_round`, so a client can tell "speculation
off" from "speculation on and not accepting anything".

## VRAM

The server refuses to start if the requested `--max-context` does not fit with a
512 MB margin, and prints the budget it computed. The drafter adds 1.25 GiB at
W4A16. At `--max-context 4096` that leaves 6.9 GiB free; at 128K the target
alone is close enough to the card that the drafter does not fit, which is a
stated limitation rather than a runtime surprise.

## Not done

* **No continuous batching.** Explicitly out of scope from the start.
* **No prefix cache** (G8). Never started.
* **Vision is deferred.** The loader detects and skips `model.visual.*`
  (0.858 GiB) and logs it; the API returns 400 on image or video content parts.
