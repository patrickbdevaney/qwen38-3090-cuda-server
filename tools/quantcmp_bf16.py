#!/usr/bin/env python3
"""
BF16 reference logits for the quantisation comparison.

This is the only place the ORIGINAL bf16 checkpoint (Qwen/Qwen3.8-27B, 52 GiB)
is used. It is a development artefact: it never fits in 24 GiB and the server
never reads it. Everything downstream consumes the logits this writes.

Not to be confused with tools/dump_reference.py, whose "BF16" input is the
dequantised AWQ INT4 weights -- that oracle answers "does our kernel match HF on
the same weights", this one answers "what did quantisation cost against the
weights Qwen actually released".

Method: teacher forcing. The whole prompt goes through the model in one forward
pass, and we keep the logits at every position, so a 13K-token prompt yields 13K
independent next-token distributions rather than one. The lm_head is applied in
chunks and each chunk is moved to host memory as fp16, because 13552 positions x
248064 vocab in fp32 is 13.4 GiB and would not co-exist with the model.

The weights are streamed layer by layer through accelerate's offload, so peak
VRAM stays under --gpu-gib and the 52 GiB is read from NVMe once.

Usage:
  python3 tools/quantcmp_bf16.py ~/qwen38-weights/bf16 \
      --prompts tests/fixtures/quantcmp/prompts.json \
      --out /home/patrickd/qwen38-weights/quantcmp/bf16
"""
import argparse, json, os, time
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--prompts", default="tests/fixtures/quantcmp/prompts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-gib", type=int, default=20)
    ap.add_argument("--cpu-gib", type=int, default=44)
    ap.add_argument("--chunk", type=int, default=512, help="lm_head positions per chunk")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

    prompts = json.load(open(a.prompts))
    tok = AutoTokenizer.from_pretrained(a.model_dir)

    offload = os.path.join(a.out, "_offload")
    os.makedirs(offload, exist_ok=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir,
        dtype = torch.bfloat16,
        device_map = "auto",
        max_memory = {0: f"{a.gpu_gib}GiB", "cpu": f"{a.cpu_gib}GiB"},
        offload_folder = offload,
        offload_state_dict = True,
    )
    model.eval()
    print("loaded in %.1f s" % (time.time() - t0), flush=True)

    manifest = {"model": a.model_dir, "dtype": "float16", "prompts": {}}
    for name, text in prompts.items():
        ids = tok(text, return_tensors = "pt").input_ids
        T = ids.shape[1]
        t1 = time.time()
        with torch.inference_mode():
            # Base model only: the full lm_head over every position at once is
            # what blows up, so take the residual stream and project in chunks.
            h = model.model(input_ids = ids.to(model.device)).last_hidden_state[0]
            head = model.lm_head
            V = head.out_features
            buf = np.empty((T, V), dtype = np.float16)
            for s in range(0, T, a.chunk):
                e = min(s + a.chunk, T)
                buf[s:e] = head(h[s:e].to(head.weight.dtype)).float().to(
                    torch.float16).cpu().numpy()
        path = os.path.join(a.out, name + ".f16")
        buf.tofile(path)
        top1 = buf.argmax(axis = 1).astype(np.int32)
        manifest["prompts"][name] = {
            "tokens": T, "vocab": int(V),
            "ids": ids[0].tolist(),
            "top1": top1.tolist(),
            "file": name + ".f16",
        }
        print("%-16s %6d tok  %6.1f s  ->  %s (%.2f GiB)"
              % (name, T, time.time() - t1, path, buf.nbytes / 2**30), flush=True)
        del h, buf
        torch.cuda.empty_cache()

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f)
    print("total %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
