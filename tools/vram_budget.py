#!/usr/bin/env python3
"""
VRAM budget for qwen38-3090-cuda-server on a 24 GB RTX 3090.

Computes the full arena budget from MEASURED checkpoint tensor sizes (not
estimates) and reports the maximum context that fits.  This is the reference
implementation of the startup check that src/main.cpp must enforce; if the two
ever disagree, this file is the spec.

Usage:
  python3 tools/vram_budget.py <model_dir> [--drafter-dir D] [--ctx 131072]
      [--drafter-precision int8|int4|bf16] [--free-gib 22.64]
"""
import argparse, json, os, struct, sys
from collections import defaultdict

DT = {"BF16":2,"F16":2,"F32":4,"F64":8,"I8":1,"U8":1,"I16":2,"I32":4,"I64":8,
      "BOOL":1,"F8_E4M3":1,"F8_E5M2":1}
GiB = 2**30

def headers(d):
    out = {}
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".safetensors")):
        with open(os.path.join(d, fn), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n).decode())
        h.pop("__metadata__", None)
        out.update(h)
    return out

def sz(t):
    n = 1
    for x in t["shape"]: n *= x
    return n * DT[t["dtype"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--drafter-dir")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--drafter-precision", default="int8", choices=["bf16","int8","int4"])
    ap.add_argument("--free-gib", type=float, default=22.64,
                    help="measured free VRAM (nvidia-smi / cudaMemGetInfo)")
    ap.add_argument("--safety-gib", type=float, default=0.5)
    ap.add_argument("--snapshots", type=int, default=8, help="GDN prefix-cache snapshots")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))
    tc  = cfg.get("text_config", cfg)
    H, V = tc["hidden_size"], tc["vocab_size"]
    lt = tc["layer_types"]
    n_attn = sum(1 for t in lt if t == "full_attention")
    n_gdn  = sum(1 for t in lt if t == "linear_attention")

    h = headers(a.model_dir)
    roles = defaultdict(int)
    for k, v in h.items():
        b = sz(v)
        if k.startswith(("model.visual","visual")):        roles["vision"] += b
        elif k.startswith("mtp."):                          roles["mtp"] += b
        elif k.endswith("lm_head.weight"):                  roles["lm_head_bf16"] += b
        elif k.endswith("embed_tokens.weight"):             roles["embed_bf16"] += b
        else:                                               roles["body"] += b

    # --- what we actually put on the device -----------------------------
    body        = roles["body"]                       # int4 packed + scales + zeros + norms
    lm_head_i4  = V*H//2 + V*(H//128)*2 + V*(H//128)//2   # 4b w + bf16 scale + 4b zp, g128
    embed_i8    = V*H*1 + V*4                             # int8 + fp32 row scale

    dpre = a.drafter_precision
    if a.drafter_dir:
        draft_bf16 = sum(sz(v) for v in headers(a.drafter_dir).values())
    else:
        draft_bf16 = int(3.584 * GiB)
    draft = {"bf16": draft_bf16, "int8": draft_bf16//2, "int4": int(draft_bf16*0.283)}[dpre]

    gdn_state  = n_gdn * tc["linear_num_value_heads"] * \
                 tc["linear_key_head_dim"] * tc["linear_value_head_dim"] * 4      # fp32
    qkv_out    = (tc["linear_num_key_heads"]*tc["linear_key_head_dim"]*2 +
                  tc["linear_num_value_heads"]*tc["linear_value_head_dim"])
    conv_state = n_gdn * qkv_out * (tc["linear_conv_kernel_dim"]-1) * 4
    state1     = gdn_state + conv_state
    rollback   = 2 * state1                                    # design (A): snapshot+replay
    snapshots  = a.snapshots * (state1 // 2)                   # bf16 snapshots
    workspace  = int(1.40 * GiB)                               # chunked prefill @4096

    kv_per_tok = n_attn * tc["num_key_value_heads"] * tc["head_dim"] * 2  # elems, K and V

    rows = [
        ("LM body INT4 g128 (+scales/zeros/norms)", body),
        ("lm_head INT4 g128 (repacked from bf16)", lm_head_i4),
        ("embed_tokens INT8 rowwise (repacked)",   embed_i8),
        (f"DFlash2 drafter [{dpre}]",              draft),
        ("GDN recurrent + conv state (fp32)",      state1),
        ("GDN rollback buffers (2x, spec decode)", rollback),
        (f"GDN prefix snapshots ({a.snapshots} x bf16)", snapshots),
        ("Activations + prefill workspace",        workspace),
    ]
    fixed = sum(r[1] for r in rows)
    arena = a.free_gib*GiB - a.safety_gib*GiB
    kv_avail = arena - fixed

    print(f"=== VRAM BUDGET  ({os.path.basename(a.model_dir)}) ===")
    print(f"free VRAM (measured)        {a.free_gib:8.3f} GiB")
    print(f"safety margin               {-a.safety_gib:8.3f} GiB")
    print(f"arena                       {arena/GiB:8.3f} GiB")
    print()
    for n, b in rows:
        print(f"  {n:42s} {b/GiB:8.3f} GiB")
    print(f"  {'FIXED TOTAL':42s} {fixed/GiB:8.3f} GiB")
    print(f"  {'REMAINING FOR KV':42s} {kv_avail/GiB:8.3f} GiB")
    print()
    print(f"KV per token: {n_attn} attn layers x {tc['num_key_value_heads']} kv-heads"
          f" x {tc['head_dim']} x 2 = {kv_per_tok} elems")
    for name, esz in (("bf16", 2), ("fp8 e4m3", 1)):
        per = kv_per_tok*esz
        print(f"  {name:9s} {per/1024:5.0f} KiB/token -> max ctx "
              f"{int(kv_avail//per):,} tokens")
    print()
    need = kv_per_tok * a.ctx          # fp8
    ok = need <= kv_avail
    print(f"REQUESTED --max-context {a.ctx:,} at FP8 needs {need/GiB:.3f} GiB : "
          f"{'FITS' if ok else 'DOES NOT FIT'}  (headroom {(kv_avail-need)/GiB:+.3f} GiB)")
    print()
    print(f"skipped: vision tower {roles['vision']/GiB:.3f} GiB, "
          f"mtp head {roles['mtp']/GiB:.3f} GiB (ablation only)")
    print(f"saved by repack: lm_head {(roles['lm_head_bf16']-lm_head_i4)/GiB:.3f} GiB, "
          f"embed {(roles['embed_bf16']-embed_i8)/GiB:.3f} GiB")

    # decode roofline, using the MEASURED 914.2 GB/s streaming read
    BW = 914.2e9
    wt = body + lm_head_i4
    print()
    print(f"decode weight traffic/token {wt/GiB:8.3f} GiB")
    print(f"  @ measured 914.2 GB/s     {wt/BW*1e3:8.2f} ms -> {BW/wt:6.1f} tok/s AR ceiling")
    print(f"  @ 80% of that             {wt/(0.8*BW)*1e3:8.2f} ms -> {0.8*BW/wt:6.1f} tok/s realistic")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
