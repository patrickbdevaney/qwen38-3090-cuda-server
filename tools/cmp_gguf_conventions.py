#!/usr/bin/env python3
"""
Check, against the TRUE BF16 checkpoint, every transform llama.cpp's converter
applies on the way into a GGUF for this model.

Why this exists. A GGUF whose conventions are not undone loads, runs at full
speed, and emits fluent nonsense -- there is no crash and no NaN to chase. The
first attempt at the loader was debugged by diffing GGUF tensors against the AWQ
checkpoint, which is the wrong oracle twice over: AWQ folds its activation-aware
per-channel scales INTO the norm that precedes each quantised linear, so
`input_layernorm` legitimately differs there, and the AWQ tensors carry INT4
error of their own. Only the original BF16 weights settle it.

Each check is elementwise and exact for the small tensors; the big projections
are compared with a loose tolerance because 3-bit quantisation error is the
floor there, and the point is the LAYOUT, not the values.

Usage:
  python3 tools/cmp_gguf_conventions.py <bf16_dir> <file.gguf> [gguf-py path]
"""
import sys, json
import numpy as np

bf16_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/patrickd/qwen38-weights/bf16"
gguf_path = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/patrickd/qwen38-weights/gguf/Qwen3.8-27B-UD-Q3_K_XL.gguf"
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3
                else "/home/patrickd/qwen38-weights/llama.cpp/gguf-py")

from gguf import GGUFReader, quants          # noqa: E402
from safetensors import safe_open            # noqa: E402

P = "model.language_model."
_idx = json.load(open(bf16_dir + "/model.safetensors.index.json"))["weight_map"]


def hf(name):
    n = P + name
    with safe_open(bf16_dir + "/" + _idx[n], framework="pt") as f:
        return f.get_tensor(n).float().numpy()


_r = GGUFReader(gguf_path)
_tt = {t.name: t for t in _r.tensors}


def gg(name):
    t = _tt[name]
    shape = tuple(reversed([int(x) for x in t.shape]))
    d = np.array(t.data)
    if t.tensor_type.name in ("F32", "F16", "BF16"):
        return d.astype(np.float32).reshape(shape)
    return quants.dequantize(d, t.tensor_type).astype(np.float32).reshape(shape)


# GdnDims for this model.
K, V, HD = 16, 48, 128
R = V // K


def ungroup(x, dim, head_dim):
    """GGUF tiled v-head order -> HF grouped order, along `dim`."""
    shape = list(x.shape)
    nd = shape[:dim] + [R, K, head_dim] + shape[dim + 1:]
    y = x.reshape(*nd)
    perm = list(range(len(nd)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return np.ascontiguousarray(y.transpose(*perm)).reshape(*shape)


fails = []


def check(tag, got, want, tol):
    got, want = got.ravel(), want.ravel()
    if got.size != want.size:
        print(f"  {tag:38s} SIZE {got.size} vs {want.size}   FAIL")
        fails.append(tag)
        return
    d = np.abs(got - want).max() / (np.abs(want).max() + 1e-9)
    ok = d <= tol
    print(f"  {tag:38s} rel {d:.3e} (tol {tol:.0e})   {'ok' if ok else 'FAIL'}")
    if not ok:
        fails.append(tag)
        print("      gguf-derived:", np.array2string(got[:6], precision=5))
        print("      bf16        :", np.array2string(want[:6], precision=5))


print("norms: ggml stores (1 + w); HF RMSNorm applies the +1 itself")
for g, h in [("blk.0.attn_norm.weight", "layers.0.input_layernorm.weight"),
             ("blk.0.post_attention_norm.weight", "layers.0.post_attention_layernorm.weight"),
             ("blk.3.attn_q_norm.weight", "layers.3.self_attn.q_norm.weight"),
             ("blk.3.attn_k_norm.weight", "layers.3.self_attn.k_norm.weight"),
             ("output_norm.weight", "norm.weight")]:
    check(g + " - 1", gg(g) - 1.0, hf(h), 0.0)

print("ssm_norm is the one norm ggml does NOT shift")
check("blk.0.ssm_norm.weight", gg("blk.0.ssm_norm.weight"),
      hf("layers.0.linear_attn.norm.weight"), 0.0)

print("ssm_a holds -exp(A_log)")
check("log(-blk.0.ssm_a), untiled", ungroup(np.log(-gg("blk.0.ssm_a")), 0, 1),
      hf("layers.0.linear_attn.A_log"), 0.0)

print("v heads: grouped in HF, tiled in the GGUF")
L = "layers.0.linear_attn."
check("ssm_dt.bias", ungroup(gg("blk.0.ssm_dt.bias"), 0, 1), hf(L + "dt_bias"), 0.0)
check("ssm_alpha.weight rows", ungroup(gg("blk.0.ssm_alpha.weight"), 0, 1),
      hf(L + "in_proj_a.weight"), 5e-3)       # Q8_0 in the file
check("ssm_beta.weight rows", ungroup(gg("blk.0.ssm_beta.weight"), 0, 1),
      hf(L + "in_proj_b.weight"), 5e-3)
conv = gg("blk.0.ssm_conv1d.weight")
check("ssm_conv1d v channels",
      np.concatenate([conv[:2 * K * HD], ungroup(conv[2 * K * HD:], 0, HD)], 0),
      hf(L + "conv1d.weight").squeeze(), 0.0)
qkv = gg("blk.0.attn_qkv.weight")
check("attn_qkv v rows",
      np.concatenate([qkv[:2 * K * HD], ungroup(qkv[2 * K * HD:], 0, HD)], 0),
      hf(L + "in_proj_qkv.weight"), 8e-2)     # 3-bit floor; layout is the point
check("attn_gate rows", ungroup(gg("blk.0.attn_gate.weight"), 0, HD),
      hf(L + "in_proj_z.weight"), 8e-2)
check("ssm_out columns", ungroup(gg("blk.0.ssm_out.weight"), 1, HD),
      hf(L + "out_proj.weight"), 8e-2)

print()
if fails:
    print("FAIL:", ", ".join(fails))
    sys.exit(1)
print("PASS: every convention in conversion/qwen.py accounted for")
