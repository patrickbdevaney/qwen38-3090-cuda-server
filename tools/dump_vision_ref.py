#!/usr/bin/env python3
"""
VISION ORACLE -- goldens for the CUDA vision tower.

Drives the OFFICIAL transformers Qwen3_5VisionModel so the CUDA tower is checked
against the reference implementation rather than against my reading of it. The
vision tower is unquantised in the checkpoint (it is in the 311-entry `ignore`
list), so the weights load straight out of the safetensors and this needs no
27B forward and no GPU.

The input is a synthetic pixel_values tensor rather than a decoded image, on
purpose: it separates "does the ViT compute the right thing" from "does our JPEG
decode and bicubic resize match PIL", which are different questions with
different tolerances.

Dumps, for one image of grid (t, h, w):
  pixel_values      [t*h*w, 3*temporal*patch*patch]
  grid_thw          [1, 3]
  stage taps        patch_embed, +pos_embed, block 0 pieces
  out               [t*h*w/(merge*merge), out_hidden]

Usage:
  python3 tools/dump_vision_ref.py <checkpoint_dir> [--out DIR] [--grid H W]
"""
import argparse, json, os, sys
import numpy as np
import torch


def w(t):
    return t.detach().to(torch.bfloat16).view(torch.uint16).cpu().numpy()


def f32(t):
    return t.detach().float().cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default="tests/fixtures/vision")
    ap.add_argument("--grid", type=int, nargs=2, default=[16, 16],
                    help="patch grid H W (must be multiples of spatial_merge_size)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from safetensors import safe_open

    cfg_all = json.load(open(os.path.join(a.ckpt, "config.json")))
    vcfg = Qwen3_5VisionConfig(**cfg_all["vision_config"])
    torch.manual_seed(1234)

    model = Qwen3_5VisionModel(vcfg)
    model.eval()

    # Load model.visual.* out of the shards.
    where = {}
    for fn in sorted(f for f in os.listdir(a.ckpt) if f.endswith(".safetensors")):
        with safe_open(os.path.join(a.ckpt, fn), framework="pt") as f:
            for k in f.keys():
                where[k] = fn
    sd = {}
    for k, fn in where.items():
        if not k.startswith("model.visual."):
            continue
        with safe_open(os.path.join(a.ckpt, fn), framework="pt") as f:
            sd[k[len("model.visual."):]] = f.get_tensor(k)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    if missing or unexpected:
        print("MISSING:", missing[:8])
        print("UNEXPECTED:", unexpected[:8])
        if missing:
            sys.exit("vision weights did not load cleanly")
    model = model.to(torch.bfloat16)

    gh, gw = a.grid
    merge = vcfg.spatial_merge_size
    assert gh % merge == 0 and gw % merge == 0, "grid must be a multiple of merge"
    gt = 1
    n_patch = gt * gh * gw
    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2

    pixel_values = (torch.randn(n_patch, patch_dim) * 0.5).to(torch.bfloat16)
    grid_thw = torch.tensor([[gt, gh, gw]], dtype=torch.long)

    stage = {}

    def cap(name):
        def hook(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            stage[name] = o.detach()
        return hook

    def cap_in(name):
        def hook(_m, inp):
            stage[name] = inp[0].detach()
        return hook

    model.patch_embed.register_forward_hook(cap("patch_embed"))
    # Input to block 0's norm1 == patch_embed + resampled pos_embed. This is the
    # only clean tap for the interpolation, and it is what catches an
    # align_corners mistake.
    model.blocks[0].norm1.register_forward_pre_hook(cap_in("post_pos_embed"))
    b0 = model.blocks[0]
    b0.norm1.register_forward_hook(cap("b0_norm1"))
    b0.attn.register_forward_hook(cap("b0_attn"))
    b0.norm2.register_forward_hook(cap("b0_norm2"))
    b0.mlp.register_forward_hook(cap("b0_mlp"))
    b0.register_forward_hook(cap("b0_out"))
    model.merger.register_forward_hook(cap("merger"))

    with torch.no_grad():
        out = model(hidden_states=pixel_values, grid_thw=grid_thw)
    # The model object's last_hidden_state is the PRE-merger [n_patch, hidden]
    # state. What actually reaches the language model is the merger's output,
    # [n_patch/merge^2, out_hidden], so that is the golden.
    out = stage["merger"]

    w(pixel_values).tofile(f"{a.out}/pixel_values.bf16")
    np.array(grid_thw[0], dtype=np.int32).tofile(f"{a.out}/grid_thw.i32")
    w(out).tofile(f"{a.out}/out.bf16")
    for k, v in stage.items():
        w(v).tofile(f"{a.out}/stage_{k}.bf16")
        print(f"  stage {k}: {tuple(v.shape)}")

    man = {
        "depth": vcfg.depth,
        "hidden_size": vcfg.hidden_size,
        "intermediate_size": vcfg.intermediate_size,
        "num_heads": vcfg.num_heads,
        "head_dim": vcfg.hidden_size // vcfg.num_heads,
        "in_channels": vcfg.in_channels,
        "patch_size": vcfg.patch_size,
        "temporal_patch_size": vcfg.temporal_patch_size,
        "spatial_merge_size": merge,
        "num_position_embeddings": vcfg.num_position_embeddings,
        "num_grid_per_side": int(vcfg.num_position_embeddings ** 0.5),
        "out_hidden_size": vcfg.out_hidden_size,
        "hidden_act": vcfg.hidden_act,
        "grid": [gt, gh, gw],
        "n_patch": n_patch,
        "patch_dim": patch_dim,
        "n_out": n_patch // (merge * merge),
        "layer_norm_eps": 1e-6,
        "out_absmax": float(out.float().abs().max()),
    }
    json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=2)
    print("grid", (gt, gh, gw), "patches", n_patch, "-> out", tuple(out.shape))
    print("|out|max %.4f" % man["out_absmax"])
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
