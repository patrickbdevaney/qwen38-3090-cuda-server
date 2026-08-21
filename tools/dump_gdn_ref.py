#!/usr/bin/env python3
"""
PHASE 3 ORACLE -- GatedDeltaNet per-primitive goldens.

Drives the ACTUAL transformers reference functions (causal_conv1d_fn,
torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule,
Qwen3_5RMSNormGated) on real layer-0 weights, and dumps every intermediate so a
C++ failure localises to one stage instead of "the layer is wrong".

Per-primitive goldens beat whole-layer goldens for exactly this reason; it is
the factoring deepseek-v4-flash used and it is better than dumping raw
activations alone.

Usage: python3 tools/dump_gdn_ref.py <bf16_model_dir> [--out tests/fixtures/gdn]
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

def w(t): return t.to(torch.bfloat16).view(torch.uint16).cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default="tests/fixtures/gdn")
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        causal_conv1d_fn, causal_conv1d_update,
        torch_recurrent_gated_delta_rule, torch_chunk_gated_delta_rule,
        Qwen3_5RMSNormGated, l2norm)
    from safetensors import safe_open

    files = sorted(f for f in os.listdir(a.model_dir) if f.endswith(".safetensors"))
    where = {}
    for fn in files:
        with safe_open(os.path.join(a.model_dir, fn), framework="pt") as f:
            for k in f.keys(): where[k] = fn
    def get(k):
        with safe_open(os.path.join(a.model_dir, where[k]), framework="pt") as f:
            return f.get_tensor(k)

    P = f"model.language_model.layers.{a.layer}.linear_attn."
    W = {n: get(P + n) for n in ("conv1d.weight", "A_log", "dt_bias", "norm.weight",
                                 "in_proj_qkv.weight", "in_proj_z.weight",
                                 "in_proj_a.weight", "in_proj_b.weight",
                                 "out_proj.weight")}

    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    NV, NK = tc["linear_num_value_heads"], tc["linear_num_key_heads"]
    DK, DV = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    K = tc["linear_conv_kernel_dim"]
    EPS = tc["rms_norm_eps"]
    HID = tc["hidden_size"]
    key_dim, val_dim = NK * DK, NV * DV
    conv_dim = key_dim * 2 + val_dim
    print(f"layer {a.layer}: conv_dim {conv_dim} key_dim {key_dim} val_dim {val_dim} "
          f"heads {NV}v/{NK}k dk {DK} dv {DV} K {K}")

    torch.manual_seed(4242)
    manifest = {"hidden": HID, "num_v_heads": NV, "num_k_heads": NK,
                "head_k": DK, "head_v": DV, "conv_k": K, "rms_eps": EPS,
                "key_dim": key_dim, "val_dim": val_dim, "conv_dim": conv_dim,
                "layer": a.layer, "cases": {}}

    norm = Qwen3_5RMSNormGated(DV, eps=EPS).to(torch.bfloat16)
    norm.weight.data = W["norm.weight"].clone()

    for name, T, seeded_state in (("T1_fresh", 1, False), ("T1_warm", 1, True),
                                  ("T8", 8, False), ("T64", 64, True),
                                  ("T512", 512, False)):
        d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
        h = (torch.randn(1, T, HID) * 0.05).to(torch.bfloat16)

        mixed = F.linear(h, W["in_proj_qkv.weight"])              # [1,T,conv_dim]
        z     = F.linear(h, W["in_proj_z.weight"])
        b     = F.linear(h, W["in_proj_b.weight"])
        aa    = F.linear(h, W["in_proj_a.weight"])

        conv_w = W["conv1d.weight"].squeeze(1)                    # [conv_dim, K]
        state0 = (torch.randn(1, conv_dim, K - 1) * 0.05).float() if seeded_state \
                 else torch.zeros(1, conv_dim, K - 1)
        cs = state0.clone().to(torch.bfloat16)
        if seeded_state:
            padded = torch.cat([cs, mixed.transpose(1, 2)], dim=-1)
            conv_out = F.conv1d(padded.to(conv_w.dtype), conv_w.unsqueeze(1),
                                None, padding=0, groups=conv_dim)[:, :, -T:]
            conv_out = F.silu(conv_out)
            conv_state1 = padded[:, :, -(K - 1):]
        else:
            conv_out = causal_conv1d_fn(mixed.transpose(1, 2), conv_w, None, activation="silu")
            padded = mixed.transpose(1, 2)
            conv_state1 = F.pad(padded, (K - 1 - min(T, K - 1), 0))[:, :, -(K - 1):] \
                          if T < K - 1 else padded[:, :, -(K - 1):]
        qkv = conv_out.transpose(1, 2)                            # [1,T,conv_dim]

        q, k, v = torch.split(qkv, [key_dim, key_dim, val_dim], dim=-1)
        q = q.reshape(1, T, NK, DK); k = k.reshape(1, T, NK, DK); v = v.reshape(1, T, NV, DV)
        beta = b.sigmoid()
        g = -W["A_log"].float().exp() * F.softplus(aa.float() + W["dt_bias"].float())
        q = q.repeat_interleave(NV // NK, dim=2)
        k = k.repeat_interleave(NV // NK, dim=2)

        S0 = (torch.randn(1, NV, DK, DV) * 0.02).float() if seeded_state \
             else torch.zeros(1, NV, DK, DV).float()
        core, S1 = torch_recurrent_gated_delta_rule(
            q, k, v, g=g, beta=beta, initial_state=S0.clone(),
            output_final_state=True, use_qk_l2norm_in_kernel=True)

        gated = norm(core.reshape(-1, DV), z.reshape(-1, DV)).reshape(1, T, val_dim)
        out = F.linear(gated, W["out_proj.weight"])

        w(h[0]).tofile(f"{d}/hidden.bf16")
        w(mixed[0]).tofile(f"{d}/mixed_qkv.bf16")
        w(z[0]).tofile(f"{d}/z.bf16")
        w(aa[0]).tofile(f"{d}/a.bf16")
        w(b[0]).tofile(f"{d}/b.bf16")
        state0[0].float().numpy().astype(np.float32).tofile(f"{d}/conv_state_in.f32")
        conv_state1[0].float().numpy().astype(np.float32).tofile(f"{d}/conv_state_out.f32")
        w(qkv[0]).tofile(f"{d}/qkv_postconv.bf16")
        g[0].float().numpy().astype(np.float32).tofile(f"{d}/g.f32")
        beta[0].float().numpy().astype(np.float32).tofile(f"{d}/beta.f32")
        S0[0].numpy().astype(np.float32).tofile(f"{d}/state_in.f32")
        S1[0].float().numpy().astype(np.float32).tofile(f"{d}/state_out.f32")
        w(core[0]).tofile(f"{d}/core_attn_out.bf16")
        w(gated[0]).tofile(f"{d}/gated.bf16")
        w(out[0]).tofile(f"{d}/out.bf16")

        manifest["cases"][name] = {
            "T": T, "seeded_state": seeded_state,
            "core_absmax": float(core.float().abs().max()),
            "state_out_absmax": float(S1.float().abs().max()),
            "out_absmax": float(out.float().abs().max()),
        }
        print(f"  {name}: T={T} seeded={seeded_state} |core|max {float(core.float().abs().max()):.4f} "
              f"|S1|max {float(S1.float().abs().max()):.4f} |out|max {float(out.float().abs().max()):.4f}")

    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(f"wrote {a.out}")

if __name__ == "__main__":
    sys.exit(main())
