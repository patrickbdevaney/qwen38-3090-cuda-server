#!/usr/bin/env python3
"""
PHASE 1 ORACLE -- per-layer activations, logits and greedy sequences.

This is the ground truth for every kernel gate in Phases 3-5. It runs ONCE and
its (small) output is committed; the CUDA server never links Python.

IMPORTANT SCOPE NOTE
--------------------
This script consumes a BF16 checkpoint materialised from the INT4 AWQ weights
(tools/dequantize_checkpoint.py). That BF16 copy is a DEVELOPMENT ARTEFACT ONLY.
It is not the serving path and the server never reads it. The server loads the
INT4 AWQ checkpoint directly into VRAM at ~13.6 GiB and computes W4A16, which is
the entire point of the project; see reports/PHASE_0.md section 4. The BF16 copy
exists because dequantising in-process needs ~53 GB of RAM and because
compressed-tensors 0.18.0 crashes on this checkpoint's decompress path.

Because the BF16 weights are exactly the dequantised INT4 weights, this oracle
is the right reference for the Phase 5 gate, which asks for a token-exact match
against HF "on the same quantized weights".

Outputs, under --out:
  manifest.json          shapes, dtypes, prompt text, token ids
  <prompt>/hidden_<L>.bf16   residual stream after layer L (L = 0..num_layers)
  <prompt>/logits.f32        final logits for every position
  <prompt>/greedy.json       greedy continuation ids and text
  <prompt>/taps.json         the DFlash2 taps (layers 5,19,33,47,61)

Usage:
  python3 tools/dump_reference.py <bf16_model_dir> --out tests/fixtures/reference
"""
import argparse, json, os, sys, time
import numpy as np
import torch

PROMPTS = {
    # short, so every hidden state fits in a few hundred KB
    "p0_factual": "The capital of France is",
    "p1_code":    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "p2_math":    "If a train travels 60 miles in 1.5 hours, its average speed is",
    "p3_cjk":     "人工智能的本质是",
    "p4_list":    "The three primary colors are",
    "p5_code2":   "public static int gcd(int a, int b) {\n    while (b != 0) {",
    "p6_reason":  "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. Therefore the ball costs",
    "p7_prose":   "The single most important property of GDDR6X memory for language model inference is",
    "p8_json":    "{\"name\": \"kernel\", \"arch\": \"sm_86\", \"params\": {",
    "p9_multi":   "Traduce al espanol: The quick brown fox jumps over the lazy dog. Respuesta:",
}
GREEDY_TOKENS = 32

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default="tests/fixtures/reference")
    ap.add_argument("--gpu-gib", type=int, default=20)
    ap.add_argument("--cpu-gib", type=int, default=34)
    ap.add_argument("--greedy-tokens", type=int, default=GREEDY_TOKENS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tk = AutoTokenizer.from_pretrained(a.model_dir)
    cfg = AutoConfig.from_pretrained(a.model_dir)
    tcfg = getattr(cfg, "text_config", cfg)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: f"{a.gpu_gib}GiB", "cpu": f"{a.cpu_gib}GiB"})
    model.eval()
    print(f"loaded in {time.time()-t0:.0f}s")
    print("device map spread:",
          len({str(v) for v in getattr(model, 'hf_device_map', {}).values()}), "devices")

    taps = [5, 19, 33, 47, 61]     # DFlash2 target_layer_ids
    manifest = {"model_dir": a.model_dir,
                "num_hidden_layers": tcfg.num_hidden_layers,
                "hidden_size": tcfg.hidden_size,
                "vocab_size": tcfg.vocab_size,
                "layer_types": list(tcfg.layer_types),
                "dflash2_taps": taps,
                "greedy_tokens": a.greedy_tokens,
                "prompts": {}}

    for name, text in PROMPTS.items():
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        ids = tk(text, return_tensors="pt").input_ids
        with torch.no_grad():
            out = model(ids.to(model.device), output_hidden_states=True, use_cache=False)
        hs = out.hidden_states           # len == num_layers + 1; [0] is the embedding
        logits = out.logits[0].float().cpu().numpy()

        for i, h in enumerate(hs):
            # numpy has no bfloat16, so reinterpret the raw 16 bits. The C++ gate
            # reads these as uint16 and widens to fp32 by shifting left 16, which
            # is exactly bf16 -> fp32 and is lossless.
            arr = h[0].to(torch.bfloat16).view(torch.uint16).cpu().numpy()
            arr.tofile(os.path.join(d, f"hidden_{i:02d}.bf16"))
        # Full logits for every position would be seq x 248,320 x 4 B, ~19 MB for a
        # 19-token prompt. Keep what the gates actually consume: the LAST position
        # in full (the greedy/KL target) and the top-1024 ids+values for every
        # position (enough for top-1 agreement and a truncated KL).
        logits[-1].astype(np.float32).tofile(os.path.join(d, "logits_last.f32"))
        K = 1024
        idx = np.argpartition(-logits, K, axis=-1)[:, :K]
        ordr = np.argsort(-np.take_along_axis(logits, idx, -1), axis=-1)
        idx = np.take_along_axis(idx, ordr, -1).astype(np.int32)
        val = np.take_along_axis(logits, idx.astype(np.int64), -1).astype(np.float32)
        idx.tofile(os.path.join(d, "logits_topk_idx.i32"))
        val.tofile(os.path.join(d, "logits_topk_val.f32"))

        # greedy continuation, the Phase 5 gate's target
        with torch.no_grad():
            gen = model.generate(ids.to(model.device), max_new_tokens=a.greedy_tokens,
                                 do_sample=False, temperature=None, top_p=None, top_k=None)
        cont = gen[0, ids.shape[1]:].cpu().tolist()
        json.dump({"prompt": text,
                   "prompt_ids": ids[0].tolist(),
                   "greedy_ids": cont,
                   "greedy_text": tk.decode(cont)},
                  open(os.path.join(d, "greedy.json"), "w"), ensure_ascii=False, indent=2)

        manifest["prompts"][name] = {
            "text": text,
            "ids": ids[0].tolist(),
            "seq_len": int(ids.shape[1]),
            "num_hidden_states": len(hs),
            "hidden_shape": [int(hs[0].shape[1]), int(hs[0].shape[2])],
            "logits_shape": list(logits.shape),
            "logits_topk": 1024,
            "top1_last": int(logits[-1].argmax()),
            "top1_last_text": tk.decode([int(logits[-1].argmax())]),
            "logits_absmax": float(np.abs(logits).max()),
        }
        print(f"  {name}: {ids.shape[1]} tok, {len(hs)} hidden states, "
              f"top1 {manifest['prompts'][name]['top1_last_text']!r}, "
              f"greedy {tk.decode(cont)[:60]!r}")

    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"wrote reference to {a.out}")

if __name__ == "__main__":
    sys.exit(main())
