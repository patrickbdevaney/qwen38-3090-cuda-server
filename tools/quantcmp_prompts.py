#!/usr/bin/env python3
"""
The prompt set for the BF16-vs-quantisation comparison.

These are deliberately NOT wikitext. The question this project has to answer is
whether a quantisation is safe for an agentic coding harness, and the failure
modes that matter there -- picking the wrong identifier, emitting a malformed
tool call, losing a fact that scrolled 30K tokens up -- do not show up in
average perplexity over prose. So the set is built from the workloads the north
star names: code generation, tool-call chains, multi-hop retrieval across a long
context, world knowledge, and structured output.

Every engine under test consumes the SAME text and must produce the same token
ids; the comparison asserts that before it compares any logits.

Written once to tests/fixtures/quantcmp/prompts.json so all four engines
(HF BF16, our AWQ, our GGUF via llama.cpp, exllamav3 EXL3) read one file.
"""
import json, os, sys

def long_repo_context(n_modules: int, needle_at: int, needle: str) -> str:
    """A synthetic repo map with one fact buried in it, then a question about
    that fact -- the multi-hop retrieval case, at whatever length we ask for."""
    out = ["# Repository map\n"]
    for i in range(n_modules):
        if i == needle_at:
            out.append(
                f"## module_{i}: auth\n"
                f"Handles session tokens. The signing key is rotated by "
                f"`rotate_key()` every {needle} seconds; nothing else in the "
                f"tree may call it.\n")
        else:
            out.append(
                f"## module_{i}: subsystem_{i}\n"
                f"Owns `Subsystem{i}` and depends on module_{max(0,i-1)}. "
                f"Exposes `run_{i}(cfg)` and `reset_{i}()`. Config key "
                f"`subsystem_{i}.enabled` defaults to false.\n")
    out.append(
        "\n# Question\n"
        "In the repository map above, how often is the auth signing key "
        "rotated, and which function performs the rotation? Answer in one "
        "sentence, citing the module.\n\n# Answer\n")
    return "".join(out)

PROMPTS = {
  "code_impl":
    "Implement a thread-safe LRU cache in Python with `get(key)` and "
    "`put(key, value)` in O(1). Use a dict plus a doubly linked list, guard "
    "every public method with a single reentrant lock, and evict the least "
    "recently used entry when `capacity` is exceeded. Include type hints and "
    "a short docstring on the class. Write only the code.\n\n"
    "```python\nimport threading\nfrom typing import Optional, Dict\n\n"
    "class LRUCache:\n",

  "code_debug":
    "The function below is supposed to return the k largest elements of "
    "`nums` in descending order, but it returns them ascending and drops "
    "duplicates. Explain both bugs and give the corrected function.\n\n"
    "```python\ndef top_k(nums, k):\n    seen = set(nums)\n"
    "    return sorted(seen)[:k]\n```\n\nThe bugs are:\n",

  "tool_call":
    "You have these tools:\n"
    "  read_file(path: str) -> str\n"
    "  list_dir(path: str) -> list[str]\n"
    "  write_file(path: str, content: str) -> None\n"
    "  run_tests(pattern: str) -> str\n\n"
    "Task: the test `tests/test_parser.py::test_nested` fails. Find the "
    "parser source, read it, fix the nesting bug, and re-run just that test. "
    "Emit one tool call per line as strict JSON with keys \"tool\" and "
    "\"args\", no prose.\n\n"
    "{\"tool\": \"list_dir\", \"args\": {\"path\": \"tests\"}}\n",

  "json_struct":
    "Convert this changelog into strict JSON matching the schema "
    "{\"version\": str, \"date\": str, \"changes\": [{\"kind\": "
    "\"added\"|\"fixed\"|\"removed\", \"text\": str}]}. Emit only JSON.\n\n"
    "v2.4.0 - 2024-11-03\n"
    " * added streaming responses to the /chat endpoint\n"
    " * fixed a crash when the prompt exceeded the context window\n"
    " * removed the deprecated --legacy-tokenizer flag\n"
    " * fixed incorrect token counts in the usage field\n\n"
    "{\n",

  "world_knowledge":
    "Answer each briefly and factually.\n"
    "1. Which memory technology do NVIDIA's RTX 3090 and 3090 Ti use, and "
    "what is the 3090's peak theoretical bandwidth?\n"
    "2. What does the CUDA compute capability sm_86 correspond to?\n"
    "3. In the Transformer paper, what is the purpose of scaling the dot "
    "product by 1/sqrt(d_k)?\n"
    "4. What distinguishes a gated delta network from standard softmax "
    "attention in terms of memory growth with sequence length?\n\n"
    "1.",

  "math_reason":
    "A batch job processes 3 shards in parallel. Shard A takes 14 minutes, "
    "shard B takes 9 minutes, and shard C takes 21 minutes. After all shards "
    "finish there is a 6-minute merge step. The job is retried from scratch "
    "if the merge fails, which happens 25% of the time, and a retry never "
    "fails. What is the expected total wall-clock time? Show the reasoning "
    "step by step, then give the number.\n\nStep 1:",

  "multihop_short":  long_repo_context(24,  9, "900"),
  "multihop_long":   long_repo_context(220, 31, "1800"),
}

def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/quantcmp/prompts.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(PROMPTS, f, indent=1, ensure_ascii=False)
    print("wrote %s: %d prompts, %d chars total"
          % (out, len(PROMPTS), sum(len(v) for v in PROMPTS.values())))

if __name__ == "__main__":
    main()
