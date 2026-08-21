#!/usr/bin/env python3
"""
PHASE 4 ORACLE -- gated attention per-primitive goldens.

Drives the actual transformers reference (Qwen3_5TextRotaryEmbedding,
apply_rotary_pos_emb, Qwen3_5RMSNorm, eager_attention_forward) on real layer
weights.

Two reference variants are dumped:
  * bf16 KV  -- the model as HF runs it; the difference against this is the COST
                of the FP8 cache, which the directive gates separately on top-1
                agreement and KL.
  * fp8  KV  -- K and V round-tripped through float8_e4m3fn before attention, so
                the C++ kernel is compared against the same information it has.
                This isolates kernel correctness from the quantization choice.

Positions deliberately include 0, 1, 4095, 32767 and 131071: mrope errors are
the directive's failure mode #1 -- fluent output that only decays at long
context -- so the table is checked where it actually matters.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

def w(t): return t.to(torch.bfloat16).view(torch.uint16).cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("--out", default="tests/fixtures/attn")
    ap.add_argument("--layer", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb, Qwen3_5RMSNorm, repeat_kv)
    from transformers import AutoConfig
    from safetensors import safe_open

    cfg = AutoConfig.from_pretrained(a.model_dir)
    tc = getattr(cfg, "text_config", cfg)
    NQ, NKV, D = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim
    EPS, HID = tc.rms_norm_eps, tc.hidden_size
    ROT = int(D * tc.partial_rotary_factor)
    print(f"layer {a.layer}: {NQ}q/{NKV}kv heads, D={D}, rot={ROT}, eps={EPS}")

    files = sorted(f for f in os.listdir(a.model_dir) if f.endswith(".safetensors"))
    where = {}
    for fn in files:
        with safe_open(os.path.join(a.model_dir, fn), framework="pt") as f:
            for k in f.keys(): where[k] = fn
    def get(k):
        with safe_open(os.path.join(a.model_dir, where[k]), framework="pt") as f:
            return f.get_tensor(k)
    P = f"model.language_model.layers.{a.layer}.self_attn."
    W = {n: get(P + n) for n in ("q_proj.weight","k_proj.weight","v_proj.weight",
                                 "o_proj.weight","q_norm.weight","k_norm.weight")}

    qn = Qwen3_5RMSNorm(D, eps=EPS).to(torch.bfloat16); qn.weight.data = W["q_norm.weight"].clone()
    kn = Qwen3_5RMSNorm(D, eps=EPS).to(torch.bfloat16); kn.weight.data = W["k_norm.weight"].clone()
    rot = Qwen3_5TextRotaryEmbedding(tc)

    manifest = {"num_q_heads": NQ, "num_kv_heads": NKV, "head_dim": D, "rotary_dim": ROT,
                "rms_eps": EPS, "hidden": HID, "layer": a.layer,
                "rope_theta": tc.rope_parameters["rope_theta"],
                "mrope_section": tc.rope_parameters["mrope_section"], "cases": {}}
    torch.manual_seed(31337)

    # position sets: contiguous prefills, plus the long-context probes
    CASES = [("T1_pos0",   [0]),
             ("T1_pos1",   [1]),
             ("T8",        list(range(8))),
             ("T128",      list(range(128))),
             ("far_4095",  [4095]),
             ("far_32767", [32767]),
             ("far_131071",[131071]),
             ("span_far",  list(range(131064, 131072)))]

    for name, positions in CASES:
        d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
        T = len(positions)
        h = (torch.randn(1, T, HID) * 0.05).to(torch.bfloat16)
        pos = torch.tensor(positions, dtype=torch.long).view(1, T)

        qg = F.linear(h, W["q_proj.weight"]).view(1, T, -1, D * 2)
        query, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(1, T, -1)
        query = qn(query.view(1, T, NQ, D)).transpose(1, 2)
        key   = kn(F.linear(h, W["k_proj.weight"]).view(1, T, NKV, D)).transpose(1, 2)
        value = F.linear(h, W["v_proj.weight"]).view(1, T, NKV, D).transpose(1, 2)

        cos, sin = rot(h, pos)
        q_r, k_r = apply_rotary_pos_emb(query, key, cos, sin)

        def attend(kk, vv):
            ks, vs = repeat_kv(kk, NQ // NKV), repeat_kv(vv, NQ // NKV)
            aw = torch.matmul(q_r, ks.transpose(2, 3)) * (D ** -0.5)
            mask = torch.full((T, T), float("-inf")).triu(1)
            aw = aw + mask
            aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q_r.dtype)
            return torch.matmul(aw, vs).transpose(1, 2).reshape(1, T, -1)

        o_bf16 = attend(k_r, value)
        k8 = k_r.to(torch.float8_e4m3fn).to(torch.bfloat16)
        v8 = value.to(torch.float8_e4m3fn).to(torch.bfloat16)
        o_fp8 = attend(k8, v8)

        gated = o_fp8 * torch.sigmoid(gate)
        out = F.linear(gated, W["o_proj.weight"])

        w(h[0]).tofile(f"{d}/hidden.bf16")
        w(qg.reshape(1, T, -1)[0]).tofile(f"{d}/qkv_q.bf16")
        w(F.linear(h, W["k_proj.weight"])[0]).tofile(f"{d}/qkv_k.bf16")
        w(F.linear(h, W["v_proj.weight"])[0]).tofile(f"{d}/qkv_v.bf16")
        np.array(positions, dtype=np.int32).tofile(f"{d}/positions.i32")
        cos[0,0].float().numpy().astype(np.float32).tofile(f"{d}/cos.f32")
        sin[0,0].float().numpy().astype(np.float32).tofile(f"{d}/sin.f32")
        w(q_r.transpose(1,2).reshape(T, NQ*D)).tofile(f"{d}/q_roped.bf16")
        w(k_r.transpose(1,2).reshape(T, NKV*D)).tofile(f"{d}/k_roped.bf16")
        w(k8.transpose(1,2).reshape(T, NKV*D)).tofile(f"{d}/k_fp8.bf16")
        w(v8.transpose(1,2).reshape(T, NKV*D)).tofile(f"{d}/v_fp8.bf16")
        w(gate[0]).tofile(f"{d}/gate.bf16")
        w(o_bf16[0]).tofile(f"{d}/attn_bf16kv.bf16")
        w(o_fp8[0]).tofile(f"{d}/attn_fp8kv.bf16")
        w(gated[0]).tofile(f"{d}/gated.bf16")
        w(out[0]).tofile(f"{d}/out.bf16")

        dq = (o_bf16.float() - o_fp8.float()).abs().max().item()
        manifest["cases"][name] = {"T": T, "positions": positions,
                                   "fp8_vs_bf16_maxabs": dq,
                                   "attn_absmax": float(o_bf16.float().abs().max())}
        print(f"  {name}: T={T} pos[0]={positions[0]} |attn|max "
              f"{float(o_bf16.float().abs().max()):.4f}  fp8-vs-bf16 maxabs {dq:.3e}")

    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=2)
    print(f"wrote {a.out}")

if __name__ == "__main__":
    sys.exit(main())
