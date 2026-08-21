#!/usr/bin/env python3
"""
PHASE 7 ORACLE -- DFlash2 drafter goldens.

Drives the OFFICIAL z-lab/dflash reference (DFlash2DraftModel) so the CUDA
drafter is checked against the spec rather than against my reading of it. The
directive is explicit: do not invent an acceptance rule and call it DFlash.

Dumps, for a synthetic context of T positions and one block of 8:
  target_hidden   [T, 5*5120]  the five tap layers concatenated in order
  fc/hidden_norm outputs
  per-layer intermediates (conv prepare/finish, attention, mlp)
  drafter output hidden [8, 5120]
  logits for the 7 mask slots (via the TARGET's lm_head)
  selector unary / candidates / chosen path

Usage:
  python3 tools/dump_dflash_ref.py <drafter_dir> <target_bf16_dir> [--out DIR]
"""
import argparse, json, os, sys
import numpy as np
import torch

DFLASH_REPO = "/tmp/claude-1000/-home-patrickd/8587d0da-43c6-48e7-89a6-8895d912822c/scratchpad/prior-art/dflash"

def w(t): return t.to(torch.bfloat16).view(torch.uint16).cpu().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drafter_dir"); ap.add_argument("target_dir")
    ap.add_argument("--out", default="tests/fixtures/dflash")
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, DFLASH_REPO)

    from dflash.model import DFlash2DraftModel, _draft_value
    from safetensors import safe_open

    DT = getattr(torch, a.dtype)
    m = DFlash2DraftModel.from_pretrained(a.drafter_dir, dtype=DT)
    m.eval()
    cfg = m.config
    H = cfg.hidden_size
    BS = m.block_size
    NT = len(m.target_layer_ids)
    print(f"drafter: {cfg.num_hidden_layers} layers, hidden {H}, block {BS}, "
          f"taps {m.target_layer_ids}, mask {m.mask_token_id}")

    # the target's embedding and lm_head -- the drafter has neither of its own
    files = sorted(f for f in os.listdir(a.target_dir) if f.endswith(".safetensors"))
    where = {}
    for fn in files:
        with safe_open(os.path.join(a.target_dir, fn), framework="pt") as f:
            for k in f.keys(): where[k] = fn
    def get(k):
        with safe_open(os.path.join(a.target_dir, where[k]), framework="pt") as f:
            return f.get_tensor(k)
    embed = get("model.language_model.embed_tokens.weight")
    lm_head = get("lm_head.weight")
    emb_scale = float(_draft_value(cfg, "input_embedding_scale", 1.0))
    print(f"input_embedding_scale {emb_scale}")

    torch.manual_seed(777)
    T = a.ctx
    target_hidden = (torch.randn(1, T, NT * H) * 0.05).to(torch.bfloat16).to(DT)
    anchor = 1000
    block_ids = torch.tensor([[anchor] + [m.mask_token_id] * (BS - 1)], dtype=torch.long)
    noise_embedding = (torch.nn.functional.embedding(block_ids, embed) * emb_scale).to(DT)
    # positions span the context then the block, as the reference slices them
    position_ids = torch.arange(T + BS).view(1, -1)[:, -(T + BS):]

    # Stage taps, so a mismatch says WHICH stage rather than just "wrong".
    stage = {}
    def cap(name):
        def hook(_mod, _inp, out):
            stage[name] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook
    def cap_in(name):
        def hook(_mod, inp):
            stage[name] = inp[0].detach()
        return hook
    m.fc.register_forward_hook(cap("fc_out"))
    m.hidden_norm.register_forward_hook(cap("ctx_norm"))
    L0 = m.layers[0]
    L0.input_layernorm.register_forward_hook(cap("l0_ln"))
    L0.self_attn.q_proj.register_forward_pre_hook(cap_in("l0_conv_prepare"))
    L0.self_attn.register_forward_hook(cap("l0_attn"))
    L0.post_attention_layernorm.register_forward_hook(cap("l0_post_ln"))
    L0.register_forward_hook(cap("l0_out"))

    with torch.no_grad():
        hidden = m(position_ids=position_ids,
                   noise_embedding=noise_embedding,
                   target_hidden=target_hidden,
                   past_key_values=None, use_cache=False)
        # the reference keeps the last BS-1 rows: slot 0 is the anchor
        draft_hidden = hidden[:, 1 - BS:, :]
        logits = torch.nn.functional.linear(draft_hidden, lm_head.to(DT))
        path, candidates, qrows = m.propose(draft_hidden,
                                            block_ids[:, 0],
                                            lambda h: torch.nn.functional.linear(h, lm_head.to(DT)),
                                            0.0)
        # the selector's inputs, for stage-level checking
        unary, cand = torch.topk(logits, m.candidate_selector.top_k, dim=-1, sorted=False)
        hp = m.candidate_selector.hidden_projection(draft_hidden)

    for k, v in stage.items():
        t = v if v.dim() == 2 else v[0]
        w(t).tofile(f"{a.out}/stage_{k}.bf16")
        print(f"  stage {k}: {tuple(t.shape)}")

    w(target_hidden[0]).tofile(f"{a.out}/target_hidden.bf16")
    w(noise_embedding[0]).tofile(f"{a.out}/noise_embedding.bf16")
    np.array(block_ids[0], dtype=np.int32).tofile(f"{a.out}/block_ids.i32")
    np.array(position_ids[0], dtype=np.int32).tofile(f"{a.out}/positions.i32")
    w(hidden[0]).tofile(f"{a.out}/hidden_out.bf16")
    w(draft_hidden[0]).tofile(f"{a.out}/draft_hidden.bf16")
    logits[0].float().numpy().astype(np.float32).tofile(f"{a.out}/logits.f32")
    unary[0].float().numpy().astype(np.float32).tofile(f"{a.out}/unary.f32")
    np.array(cand[0], dtype=np.int32).tofile(f"{a.out}/candidates.i32")
    w(hp[0]).tofile(f"{a.out}/hidden_proj.bf16")
    np.array(path[0], dtype=np.int32).tofile(f"{a.out}/path.i32")

    man = {"hidden": H, "block_size": BS, "num_layers": cfg.num_hidden_layers,
           "num_taps": NT, "target_layer_ids": list(m.target_layer_ids),
           "mask_token_id": int(m.mask_token_id), "ctx": T,
           "top_k": int(m.candidate_selector.top_k),
           "rank": int(m.candidate_selector.predecessor_codebook.weight.shape[1]),
           "input_embedding_scale": emb_scale,
           "sliding_window": int(cfg.sliding_window),
           "num_q_heads": int(cfg.num_attention_heads),
           "num_kv_heads": int(cfg.num_key_value_heads),
           "head_dim": int(cfg.head_dim),
           "rms_eps": float(cfg.rms_norm_eps),
           "rope_theta": float(cfg.rope_parameters["rope_theta"]),
           "conv_kernel_size": int(_draft_value(cfg, "conv_kernel_size")),
           "conv_group_size": int(_draft_value(cfg, "conv_group_size")),
           "path": [int(x) for x in path[0]],
           "hidden_absmax": float(hidden.float().abs().max()),
           "logits_absmax": float(logits.float().abs().max())}
    json.dump(man, open(f"{a.out}/manifest.json", "w"), indent=2)
    print("path:", man["path"])
    print("|hidden|max %.4f  |logits|max %.4f" % (man["hidden_absmax"], man["logits_absmax"]))
    print("wrote", a.out)

if __name__ == "__main__":
    sys.exit(main())
