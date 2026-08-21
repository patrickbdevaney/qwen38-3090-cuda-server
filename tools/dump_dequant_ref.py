#!/usr/bin/env python3
"""
PHASE 1 ORACLE -- W4A16 dequantization ground truth.

Pins the exact bit order of compressed-tensors 'pack-quantized' int4 so the
CUDA unpack cannot drift. Uses the library's OWN unpack_from_int32 as the
reference rather than a reimplementation, then writes fp32 sub-blocks that the
C++ gate compares against.

Format (derived from compressed_tensors/compressors/pack_quantized/helpers.py):
  - weights are int8 in [-8, 7], stored as nibble = value + 8
  - packing is dense little-endian along the packed dim: element i occupies
    global bits [4i, 4i+4). Since 4 divides 32, element e of a row lives in
    word e/8 at bit 4*(e%8), and nothing straddles a word boundary.
  - weight_packed uses packed_dim=1 (input features)
  - weight_zero_point uses packed_dim=0 (output features), hence its [out/8, G]
    shape, and its values carry the same -8 offset
  - dequant is  w = (q - zero_point) * scale,  per (output row, group) 

Usage: python3 tools/dump_dequant_ref.py <model_dir> [--out tests/fixtures/dequant]
"""
import argparse, json, os, struct, sys
import numpy as np
import torch
from safetensors import safe_open
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32

# Tensors chosen to cover every distinct shape class in the model.
PICKS = [
    "model.language_model.layers.0.mlp.gate_proj",          # [17408, 5120]
    "model.language_model.layers.0.mlp.down_proj",          # [5120, 17408]
    "model.language_model.layers.3.self_attn.q_proj",       # [12288, 5120]
    "model.language_model.layers.3.self_attn.k_proj",       # [1024, 5120]
    "model.language_model.layers.0.linear_attn.in_proj_qkv", # [10240, 5120]
    "model.language_model.layers.0.linear_attn.in_proj_z",   # [6144, 5120]
    "model.language_model.layers.0.linear_attn.out_proj",    # [5120, 6144]
]
ROWS, COLS = 64, 512      # sub-block dumped per tensor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default="tests/fixtures/dequant")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    files = sorted(f for f in os.listdir(a.model_dir) if f.endswith(".safetensors"))
    where = {}
    for fn in files:
        with safe_open(os.path.join(a.model_dir, fn), framework="pt") as f:
            for k in f.keys(): where[k] = fn

    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    q = cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config")
    gs = q["config_groups"]["group_0"]["weights"]["group_size"]
    nb = q["config_groups"]["group_0"]["weights"]["num_bits"]
    print(f"num_bits={nb} group_size={gs}")

    manifest = []
    for base in PICKS:
        kp, ks, kz = base + ".weight_packed", base + ".weight_scale", base + ".weight_zero_point"
        if kp not in where:
            print(f"  skip (absent): {base}"); continue
        def load(k):
            with safe_open(os.path.join(a.model_dir, where[k]), framework="pt") as f:
                return f.get_tensor(k)
        packed, scale, zp = load(kp), load(ks), load(kz)
        out_f = scale.shape[0]
        in_f  = scale.shape[1] * gs

        # Reference unpack, from the library itself.
        qw = unpack_from_int32(packed, nb, torch.Size([out_f, in_f]), packed_dim=1).to(torch.int32)
        zpu = unpack_from_int32(zp, nb, torch.Size([out_f, scale.shape[1]]), packed_dim=0).to(torch.int32)

        r, c = min(ROWS, out_f), min(COLS, in_f)
        qs   = qw[:r, :c]
        zs   = zpu[:r, : (c + gs - 1) // gs]
        ss   = scale[:r, : (c + gs - 1) // gs].float()
        gidx = torch.arange(c) // gs
        deq  = (qs - zs[:, gidx]).float() * ss[:, gidx]

        stem = base.replace("model.language_model.", "").replace(".", "_")
        deq.numpy().astype(np.float32).tofile(os.path.join(a.out, stem + ".f32"))
        qs.numpy().astype(np.int8).tofile(os.path.join(a.out, stem + ".q8"))
        manifest.append({
            "name": base, "out_features": int(out_f), "in_features": int(in_f),
            "group_size": int(gs), "num_bits": int(nb),
            "rows": int(r), "cols": int(c),
            "packed_shape": list(packed.shape), "scale_shape": list(scale.shape),
            "zp_shape": list(zp.shape),
            "deq_file": stem + ".f32", "q_file": stem + ".q8",
            "deq_absmax": float(deq.abs().max()),
            "q_min": int(qs.min()), "q_max": int(qs.max()),
            "zp_min": int(zs.min()), "zp_max": int(zs.max()),
        })
        print(f"  {base}: [{out_f}x{in_f}] q in [{int(qs.min())},{int(qs.max())}] "
              f"zp in [{int(zs.min())},{int(zs.max())}] |deq|max {float(deq.abs().max()):.4f}")

    json.dump({"group_size": gs, "num_bits": nb, "tensors": manifest},
              open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(f"wrote {len(manifest)} tensors -> {a.out}")

if __name__ == "__main__":
    sys.exit(main())
