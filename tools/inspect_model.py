#!/usr/bin/env python3
"""
GATE 0 — ground-truth verification for qwen38-3090-cuda-server.

Derives every shape the CUDA server depends on from the checkpoint itself
(config.json + safetensors headers) and asserts it against the architecture
table in the project directive.  Nothing downstream is allowed to hardcode a
shape that is not printed here.

Usage:
    python3 tools/inspect_model.py <model_dir> [--json out.json]

Reads safetensors headers only (first few hundred KB per shard); never loads
weights.  Works on a local directory of any of the AWQ/GPTQ/BF16 checkpoints.
"""
import argparse, json, os, re, struct, sys
from collections import defaultdict

# ---------------------------------------------------------------- expectations
# From the official Qwen3.8-27B card, as transcribed into the project directive.
# Every one of these is ASSERTED against the checkpoint.
EXPECT = {
    "hidden_size":            5120,
    "vocab_size":             248320,
    "num_hidden_layers":      64,
    "num_gdn_layers":         48,
    "num_attn_layers":        16,
    "linear_num_value_heads": 48,
    "linear_num_key_heads":   16,
    "linear_key_head_dim":    128,
    "linear_value_head_dim":  128,
    "num_attention_heads":    24,
    "num_key_value_heads":    4,
    "head_dim":               256,
    "partial_rotary_factor":  0.25,
    "rotary_dims":            64,
    "mrope_section":          [11, 11, 10],
    "mrope_interleaved":      True,
    "rope_theta":             10000000,
    "intermediate_size":      17408,
    "max_position_embeddings": 262144,
    "full_attention_interval": 4,
    "linear_conv_kernel_dim": 4,
}

DTYPE_BYTES = {"BF16":2,"F16":2,"F32":4,"F64":8,"I8":1,"U8":1,
               "I16":2,"I32":4,"I64":8,"BOOL":1,"F8_E4M3":1,"F8_E5M2":1}

FAILURES, WARNINGS = [], []

def check(name, got, want, note=""):
    ok = (got == want)
    mark = "OK  " if ok else "FAIL"
    if not ok:
        FAILURES.append(f"{name}: got {got!r}, directive says {want!r}")
    extra = f"   {note}" if note else ""
    print(f"  [{mark}] {name:28s} = {str(got):<24s} (expect {want}){extra}")
    return ok

def warn(msg):
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")

# ---------------------------------------------------------------- safetensors
def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n).decode())
    h.pop("__metadata__", None)
    return h

def load_all_headers(model_dir):
    """Merge headers of every shard.  Returns {name: {dtype, shape}}."""
    shards = sorted(p for p in os.listdir(model_dir) if p.endswith(".safetensors"))
    if not shards:
        sys.exit(f"no .safetensors in {model_dir}")
    out = {}
    for s in shards:
        for k, v in read_header(os.path.join(model_dir, s)).items():
            out[k] = {"dtype": v["dtype"], "shape": v["shape"], "shard": s}
    return out, shards

def nbytes(t):
    n = 1
    for d in t["shape"]: n *= d
    return n * DTYPE_BYTES[t["dtype"]]

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--json", help="write derived ModelShape to this path")
    a = ap.parse_args()
    md = a.model_dir

    cfg = json.load(open(os.path.join(md, "config.json")))
    tc  = cfg.get("text_config", cfg)          # VL wrapper or flat
    quant = cfg.get("quantization_config") or tc.get("quantization_config")

    print(f"=== GATE 0: {md} ===")
    print(f"architectures      : {cfg.get('architectures')}")
    print(f"model_type         : {cfg.get('model_type')} / text {tc.get('model_type')}")
    print(f"transformers_version: {cfg.get('transformers_version')}")
    print()

    # ---------------- 1. architecture from config.json --------------------
    print("--- 1. architecture (config.json vs directive) ---")
    layer_types = tc["layer_types"]
    n_gdn  = sum(1 for t in layer_types if t == "linear_attention")
    n_attn = sum(1 for t in layer_types if t == "full_attention")
    rope = tc["rope_parameters"]
    prf  = tc["partial_rotary_factor"]
    rot  = int(tc["head_dim"] * prf)

    check("hidden_size",            tc["hidden_size"],            EXPECT["hidden_size"])
    check("vocab_size",             tc["vocab_size"],             EXPECT["vocab_size"])
    check("num_hidden_layers",      tc["num_hidden_layers"],      EXPECT["num_hidden_layers"])
    check("num_gdn_layers",         n_gdn,                        EXPECT["num_gdn_layers"])
    check("num_attn_layers",        n_attn,                       EXPECT["num_attn_layers"])
    check("full_attention_interval",tc["full_attention_interval"],EXPECT["full_attention_interval"])
    check("linear_num_value_heads", tc["linear_num_value_heads"], EXPECT["linear_num_value_heads"])
    check("linear_num_key_heads",   tc["linear_num_key_heads"],   EXPECT["linear_num_key_heads"])
    check("linear_key_head_dim",    tc["linear_key_head_dim"],    EXPECT["linear_key_head_dim"])
    check("linear_value_head_dim",  tc["linear_value_head_dim"],  EXPECT["linear_value_head_dim"])
    check("linear_conv_kernel_dim", tc["linear_conv_kernel_dim"], EXPECT["linear_conv_kernel_dim"])
    check("num_attention_heads",    tc["num_attention_heads"],    EXPECT["num_attention_heads"])
    check("num_key_value_heads",    tc["num_key_value_heads"],    EXPECT["num_key_value_heads"])
    check("head_dim",               tc["head_dim"],               EXPECT["head_dim"])
    check("partial_rotary_factor",  prf,                          EXPECT["partial_rotary_factor"])
    check("rotary_dims",            rot,                          EXPECT["rotary_dims"],
          note="-> 192 dims pass through UNROTATED (directive 8.2)")
    check("intermediate_size",      tc["intermediate_size"],      EXPECT["intermediate_size"])
    check("max_position_embeddings",tc["max_position_embeddings"],EXPECT["max_position_embeddings"])
    check("rope_theta",             rope["rope_theta"],           EXPECT["rope_theta"])
    check("mrope_section",          rope["mrope_section"],        EXPECT["mrope_section"])
    check("mrope_interleaved",      rope.get("mrope_interleaved"),EXPECT["mrope_interleaved"])

    # mrope sections must tile the rotary half-dim exactly
    half = rot // 2
    if sum(rope["mrope_section"]) != half:
        FAILURES.append(f"mrope_section sums to {sum(rope['mrope_section'])}, expected {half} "
                        f"(= rotary_dims/2)")
        print(f"  [FAIL] mrope_section sum       = {sum(rope['mrope_section'])} (expect {half})")
    else:
        print(f"  [OK  ] mrope_section sum       = {half} == rotary_dims/2   (tiles exactly)")

    # layer pattern must be the documented 3 GDN : 1 attention repeat
    pat = "".join("A" if t == "full_attention" else "L" for t in layer_types)
    expect_pat = "LLLA" * 16
    check("layer pattern", pat == expect_pat, True, note="16 x (L,L,L,A)")
    print(f"         attention layers at indices: "
          f"{[i for i,t in enumerate(layer_types) if t=='full_attention']}")

    # ---------------- 2. the budget-critical questions ---------------------
    print("\n--- 2. budget-critical facts (directive Gate 0 / failure mode #4) ---")
    tied = cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings"))
    print(f"  tie_word_embeddings        = {tied}")
    if not tied:
        print("         -> lm_head is a SEPARATE 248320x5120 tensor. Budget it.")

    print(f"  attn_output_gate           = {tc.get('attn_output_gate')}  (gated attention)")
    print(f"  output_gate_type           = {tc.get('output_gate_type')}")
    print(f"  mamba_ssm_dtype            = {tc.get('mamba_ssm_dtype')}  (GDN state accum dtype)")
    print(f"  mtp_num_hidden_layers      = {tc.get('mtp_num_hidden_layers')}  (built-in MTP head)")
    print(f"  rms_norm_eps               = {tc.get('rms_norm_eps')}")
    if quant:
        print(f"  quantization_config        = {json.dumps(quant)[:300]}")
    else:
        print("  quantization_config        = (none in config.json)")

    # ---------------- 3. tensor manifest ----------------------------------
    print("\n--- 3. safetensors manifest ---")
    hdr, shards = load_all_headers(md)
    print(f"  shards: {len(shards)}   tensors: {len(hdr)}")

    pats = defaultdict(lambda: [0, None])
    for k, v in hdr.items():
        p = re.sub(r"\.\d+\.", ".N.", k)
        pats[p][0] += 1
        pats[p][1] = v
    for p in sorted(pats):
        c, v = pats[p]
        print(f"   {c:5d}  {p:66s} {v['dtype']:8s} {v['shape']}")

    # ---------------- 4. QK-norm / gating presence ------------------------
    print("\n--- 4. presence checks the directive asked for ---")
    has = lambda sub: any(sub in k for k in hdr)
    for name, sub, why in [
        ("attention q_norm", ".self_attn.q_norm.weight", "QK-norm on Q"),
        ("attention k_norm", ".self_attn.k_norm.weight", "QK-norm on K"),
        ("GDN conv1d",       ".linear_attn.conv1d.weight", "short conv"),
        ("GDN A_log",        ".linear_attn.A_log", "decay"),
        ("GDN dt_bias",      ".linear_attn.dt_bias", "timestep bias"),
        ("GDN in_proj_z",    ".linear_attn.in_proj_z", "output gate"),
        ("GDN norm",         ".linear_attn.norm.weight", "per-head RMSNorm"),
        ("MTP head",         "mtp.", "built-in multi-token-prediction head"),
        ("vision tower",     "model.visual.", "DEFERRED in v1 - loader must skip"),
    ]:
        print(f"  [{'yes' if has(sub) else 'NO ':3s}] {name:20s} {why}")

    # ---------------- 5. VRAM accounting ----------------------------------
    print("\n--- 5. VRAM accounting (on-disk, by role) ---")
    roles = defaultdict(int)
    for k, v in hdr.items():
        b = nbytes(v)
        if   k.startswith("model.visual") or k.startswith("visual"): roles["vision (SKIPPED)"] += b
        elif k.startswith("mtp."):                                   roles["mtp head"] += b
        elif k == "lm_head.weight" or k.endswith("lm_head.weight"):  roles["lm_head"] += b
        elif k.endswith("embed_tokens.weight"):                      roles["embed_tokens"] += b
        elif k.endswith(("weight_packed","qweight")):                roles["body int4 packed"] += b
        elif k.endswith(("weight_scale","scales")):                  roles["body scales"] += b
        elif k.endswith(("weight_zero_point","qzeros")):             roles["body zeros"] += b
        elif k.endswith("weight_shape"):                             roles["quant meta"] += b
        else:                                                        roles["body bf16 (norms etc)"] += b
    tot = sum(roles.values())
    for r in sorted(roles, key=lambda x: -roles[x]):
        print(f"  {r:26s} {roles[r]/2**30:8.3f} GiB")
    print(f"  {'TOTAL on disk':26s} {tot/2**30:8.3f} GiB")

    # effective bits-per-weight of the quantized body
    qp = 0
    for k, v in hdr.items():
        if k.endswith("weight_packed") and v["dtype"] == "I32":
            qp += v["shape"][0]*v["shape"][1]*8            # 8 int4 per int32
        elif k.endswith("qweight") and v["dtype"] == "I32":
            qp += v["shape"][0]*v["shape"][1]*8
    if qp:
        qb = roles["body int4 packed"]+roles["body scales"]+roles["body zeros"]
        print(f"\n  quantized params : {qp/1e9:.3f} B")
        print(f"  effective bpw    : {qb*8/qp:.3f}")

    # group size, from any scale tensor
    gs = None
    for k, v in hdr.items():
        if k.endswith("mlp.gate_proj.weight_scale") or k.endswith("mlp.gate_proj.scales"):
            gs = tc["hidden_size"] // v["shape"][1] if v["shape"][1] else None
            break
    if gs: print(f"  quant group size : {gs}")

    # ---------------- 6. derived runtime constants ------------------------
    print("\n--- 6. derived runtime constants (the server must use THESE) ---")
    kv_elems = n_attn * tc["num_key_value_heads"] * tc["head_dim"] * 2
    print(f"  KV elements/token        : {n_attn} layers x {tc['num_key_value_heads']} kv-heads"
          f" x {tc['head_dim']} dim x 2 = {kv_elems}")
    print(f"  KV bytes/token  bf16     : {kv_elems*2/1024:.0f} KiB")
    print(f"  KV bytes/token  fp8 e4m3 : {kv_elems*1/1024:.0f} KiB")
    for ctx in (32768, 131072, 262144):
        print(f"    ctx {ctx:7d}: bf16 {kv_elems*2*ctx/2**30:6.2f} GiB   "
              f"fp8 {kv_elems*ctx/2**30:6.2f} GiB")
    st = n_gdn*tc["linear_num_value_heads"]*tc["linear_key_head_dim"]*tc["linear_value_head_dim"]
    print(f"  GDN recurrent state      : {n_gdn} x {tc['linear_num_value_heads']} x "
          f"{tc['linear_key_head_dim']} x {tc['linear_value_head_dim']} = {st} elems")
    print(f"    fp32 {st*4/2**20:.1f} MiB   bf16 {st*2/2**20:.1f} MiB   "
          f"(CONSTANT in context length)")
    qkv_out = (tc["linear_num_key_heads"]*tc["linear_key_head_dim"]*2
               + tc["linear_num_value_heads"]*tc["linear_value_head_dim"])
    conv_st = n_gdn*qkv_out*(tc["linear_conv_kernel_dim"]-1)
    print(f"  GDN short-conv state     : {n_gdn} x {qkv_out} x "
          f"{tc['linear_conv_kernel_dim']-1} = {conv_st} elems "
          f"({conv_st*4/2**20:.1f} MiB fp32)")

    # ---------------- verdict ---------------------------------------------
    print("\n=== VERDICT ===")
    for w in WARNINGS: print(f"  WARN: {w}")
    if FAILURES:
        print(f"  GATE 0 FAILED — {len(FAILURES)} mismatch(es):")
        for f in FAILURES: print(f"    - {f}")
    else:
        print("  GATE 0 PASSED — checkpoint matches the directive's architecture table.")

    if a.json:
        shape = {k: (tc.get(k) if k in tc else EXPECT[k]) for k in EXPECT}
        shape.update({"num_gdn_layers": n_gdn, "num_attn_layers": n_attn,
                      "rotary_dims": rot, "tie_word_embeddings": tied,
                      "layer_types": layer_types,
                      "kv_elems_per_token": kv_elems,
                      "gdn_state_elems": st, "gdn_conv_state_elems": conv_st,
                      "quant_group_size": gs})
        json.dump(shape, open(a.json, "w"), indent=2)
        print(f"  wrote {a.json}")
    return 1 if FAILURES else 0

if __name__ == "__main__":
    sys.exit(main())
