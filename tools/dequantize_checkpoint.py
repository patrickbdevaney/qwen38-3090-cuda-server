#!/usr/bin/env python3
"""
Write a BF16 copy of the W4A16 checkpoint, so the HF reference oracle can be run
without a 53 GB in-process decompression.

Why this exists rather than just loading the compressed checkpoint:
  1. compressed-tensors 0.18.0 has a bug in its decompress path -- it unpacks
     weight_zero_point once (int32 -> int8) and then tries again, raising
     "Expected torch.int32 but got torch.int8". Both run_compressed=True and
     the default path hit it.
  2. Even without the bug, decompressing in-process needs ~53 GB of RAM on a
     60 GB box that is also running a desktop. Streaming to disk keeps peak
     memory at one tensor.

Dequant uses the SAME formula that tests/gate_dequant.cpp proved bit-exact
against compressed-tensors: w = (q - zp) * scale, nibble = value + 8, weights
packed along dim 1 and zero points along dim 0.

The vision tower is dropped: v1 is text-only and it is 0.86 GiB of dead weight.

Usage: python3 tools/dequantize_checkpoint.py <src_dir> <dst_dir> [--keep-vision]
"""
import argparse, gc, json, os, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def unpack_i4(packed: torch.Tensor, elems: int, dim: int) -> torch.Tensor:
    """Dense little-endian int4 unpack; returns int8 in [-8, 7]."""
    if dim == 0:
        packed = packed.t().contiguous()
    rows, words = packed.shape
    p = packed.to(torch.int32)
    sh = torch.arange(8, device=p.device, dtype=torch.int32) * 4
    n = ((p.unsqueeze(-1) >> sh) & 0xF).reshape(rows, words * 8)
    n = n[:, :elems] - 8
    out = n.to(torch.int8)
    return out.t().contiguous() if dim == 0 else out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--keep-vision", action="store_true")
    ap.add_argument("--shard-gb", type=float, default=4.0)
    a = ap.parse_args()
    os.makedirs(a.dst, exist_ok=True)

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    q = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config")
    gs = q["config_groups"]["group_0"]["weights"]["group_size"]
    nb = q["config_groups"]["group_0"]["weights"]["num_bits"]
    assert nb == 4, nb
    print(f"group_size={gs} num_bits={nb}")

    files = sorted(f for f in os.listdir(a.src) if f.endswith(".safetensors"))
    where = {}
    for fn in files:
        with safe_open(os.path.join(a.src, fn), framework="pt") as f:
            for k in f.keys(): where[k] = fn
    print(f"{len(where)} tensors across {len(files)} shards")

    # group quantized triples by their base name
    bases = sorted({k[: -len(".weight_packed")] for k in where if k.endswith(".weight_packed")})
    baseset = set(bases)
    plain = [k for k in where
             if not any(k.endswith(s) for s in
                        (".weight_packed", ".weight_scale", ".weight_zero_point", ".weight_shape"))]
    print(f"{len(bases)} quantized modules, {len(plain)} plain tensors")

    handles = {fn: safe_open(os.path.join(a.src, fn), framework="pt") for fn in files}
    def get(k): return handles[where[k]].get_tensor(k)

    shard, shard_bytes, idx, nshard = {}, 0, {}, 0
    limit = int(a.shard_gb * 1e9)
    t0 = time.time()

    def flush():
        nonlocal shard, shard_bytes, nshard
        if not shard: return
        nshard += 1
        name = f"model-{nshard:05d}.safetensors"
        save_file(shard, os.path.join(a.dst, name), metadata={"format": "pt"})
        for k in shard: idx[k] = name
        print(f"  wrote {name}  {shard_bytes/1e9:.2f} GB  ({len(shard)} tensors, "
              f"{time.time()-t0:.0f}s)")
        shard = {}; shard_bytes = 0
        gc.collect()

    def add(k, t):
        nonlocal shard_bytes
        shard[k] = t
        shard_bytes += t.numel() * t.element_size()
        if shard_bytes >= limit: flush()

    skipped_vision = 0
    for k in plain:
        if not a.keep_vision and k.startswith(("model.visual", "visual")):
            skipped_vision += 1; continue
        add(k, get(k).contiguous())

    for i, base in enumerate(bases):
        if not a.keep_vision and base.startswith(("model.visual", "visual")):
            skipped_vision += 1; continue
        packed = get(base + ".weight_packed")
        scale  = get(base + ".weight_scale").float()
        zp     = get(base + ".weight_zero_point")
        out_f, G = scale.shape
        in_f = G * gs
        qw  = unpack_i4(packed, in_f, dim=1).to(torch.int32)
        zpu = unpack_i4(zp, out_f, dim=0).to(torch.int32)          # [out, G]
        w = (qw.view(out_f, G, gs) - zpu.unsqueeze(-1)).to(torch.float32)
        w = (w * scale.unsqueeze(-1)).view(out_f, in_f).to(torch.bfloat16)
        add(base + ".weight", w.contiguous())
        del packed, scale, zp, qw, zpu, w
        if (i + 1) % 100 == 0:
            print(f"  dequantized {i+1}/{len(bases)}  ({time.time()-t0:.0f}s)")
    flush()

    json.dump({"metadata": {"total_size": sum(
                   os.path.getsize(os.path.join(a.dst, f))
                   for f in os.listdir(a.dst) if f.endswith(".safetensors"))},
               "weight_map": idx},
              open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)

    # config without the quantization block, and without vision unless kept
    cfg2 = json.loads(json.dumps(cfg))
    cfg2.pop("quantization_config", None)
    cfg2.get("text_config", {}).pop("quantization_config", None)
    if not a.keep_vision:
        cfg2["language_model_only"] = True
    json.dump(cfg2, open(os.path.join(a.dst, "config.json"), "w"), indent=2)
    for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json",
              "chat_template.jinja", "vocab.json", "merges.txt"):
        p = os.path.join(a.src, f)
        if os.path.exists(p): shutil.copy(p, a.dst)

    print(f"DONE  {nshard} shards, {len(idx)} tensors, vision tensors skipped: {skipped_vision}")
    print(f"elapsed {time.time()-t0:.0f}s")

if __name__ == "__main__":
    sys.exit(main())
