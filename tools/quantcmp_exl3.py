#!/usr/bin/env python3
"""
Teacher-forced logits from an EXL3 checkpoint, in the same fp16 [T, vocab]
layout tools/quantcmp_bf16.py and tools/quantcmp_dump.cu write, so
tools/quantcmp_score.py can score all three against the same reference.

Runs through exllamav3's own Model/Cache, chunked over the prompt so a 13K
position x 248K vocab logits tensor never has to exist on the device at once.

NOTE the construction order: the Cache must exist BEFORE model.load(), because
loading is what materialises the GatedDeltaNet recurrent state. Build it after
and every GDN layer's conv_state stays on the meta device, and the conv1d
kernel dereferences it -- see reports/EXL3.md.

Usage:
  python3 tools/quantcmp_exl3.py <exl3_dir> --prompts ... --out ...
"""
import argparse, json, os, time
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--prompts", default = "tests/fixtures/quantcmp/prompts.json")
    ap.add_argument("--out", required = True)
    ap.add_argument("--chunk", type = int, default = 512)
    ap.add_argument("--max-ctx", type = int, default = 16384)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok = True)

    from exllamav3 import Config, Model, Cache, Tokenizer

    prompts = json.load(open(a.prompts))
    cfg = Config.from_directory(a.model_dir)
    model = Model.from_config(cfg)
    tokenizer = Tokenizer.from_config(cfg)
    cache = Cache(model, max_num_tokens = a.max_ctx)     # BEFORE load, see above
    model.load()

    manifest = {"model": a.model_dir, "dtype": "float16", "prompts": {}}
    for name, text in prompts.items():
        ids = tokenizer.encode(text, add_bos = False)   # HF adds no BOS for Qwen; ids must match exactly
        T = ids.shape[-1]
        path = os.path.join(a.out, name + ".f16")
        t0 = time.time()
        base = {"attn_mode": "flash_attn", "cache": cache,
                "batch_shape": (1, a.max_ctx), "reconstruct": False}
        params = dict(base)
        with open(path, "wb") as f, torch.inference_mode():
            for s in range(0, T, a.chunk):
                e = min(s + a.chunk, T)
                params["past_len"] = s
                out = model.forward(input_ids = ids[:, s:e].cuda(), params = params)
                assert out.shape[1] == e - s, \
                    "expected logits for every position, got %s for %d" % (out.shape, e - s)
                f.write(out[0].to(torch.float16).cpu().numpy().tobytes())
        manifest["prompts"][name] = {
            "tokens": T, "vocab": int(out.shape[-1]),
            "ids": ids[0].tolist(), "file": name + ".f16"}
        print("%-16s %6d tok  %6.1f s  -> %s" % (name, T, time.time() - t0, path), flush = True)
        # Each prompt starts from past_len 0 with its own params dict, so
        # prepare_for_recurrence hands it a fresh recurrent slot and the KV
        # pages are simply overwritten. The cache defaults to 16 slots and
        # there are 8 prompts, so nothing needs releasing.
        assert len(prompts) <= cache.num_slots, \
            "%d prompts but only %d recurrent slots" % (len(prompts), cache.num_slots)

    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"))
    print("done")


if __name__ == "__main__":
    main()
