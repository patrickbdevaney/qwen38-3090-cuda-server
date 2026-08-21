#!/usr/bin/env python3
"""
Score a quantisation against the BF16 reference.

Reads two directories written by tools/quantcmp_bf16.py (reference) and any of
the candidate dumpers, each holding fp16 [T, vocab] logits per prompt plus a
manifest with the token ids. Reports, per prompt and in total:

  KL(P||Q)     mean, median and p99 of the per-position KL divergence, in nats,
               with P the BF16 distribution. The tail matters more than the
               mean: an agent breaks on the one position where the model was
               confident and the quantisation was not, and that position is
               invisible in an average.
  top-1        fraction of positions where the argmax agrees. This is the
               closest proxy for "would greedy decoding have diverged here".
  top-5 recall fraction of BF16 top-1 tokens still in the candidate's top 5,
               which is what matters under sampling rather than greedy.

Both sides are read in row chunks so a 13K x 248K fp16 file never lands in RAM
whole.

Usage:
  python3 tools/quantcmp_score.py <ref_dir> <cand_dir> [--label NAME]
"""
import argparse, json, os, sys
import numpy as np


def softmax_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.max(axis = 1, keepdims = True)
    np.exp(x, out = x)
    x /= x.sum(axis = 1, keepdims = True)
    return x


def score_prompt(ref_path, cand_path, T, V, chunk = 256):
    kls, agree, top5 = [], 0, 0
    ref = np.memmap(ref_path, dtype = np.float16, mode = "r", shape = (T, V))
    cand = np.memmap(cand_path, dtype = np.float16, mode = "r", shape = (T, V))
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        p = softmax_rows(np.asarray(ref[s:e]))
        q = softmax_rows(np.asarray(cand[s:e]))
        # KL(P||Q) = sum p (log p - log q); clamp q so a hard zero cannot make
        # the whole run inf and hide every real number behind it.
        np.maximum(q, 1e-10, out = q)
        kl = (p * (np.log(np.maximum(p, 1e-10)) - np.log(q))).sum(axis = 1)
        kls.append(kl)
        pa = p.argmax(axis = 1)
        agree += int((pa == q.argmax(axis = 1)).sum())
        # top-5 of the candidate, is the BF16 argmax in it
        idx = np.argpartition(-q, 5, axis = 1)[:, :5]
        top5 += int((idx == pa[:, None]).any(axis = 1).sum())
    kl = np.concatenate(kls)
    return kl, agree, top5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_dir")
    ap.add_argument("cand_dir")
    ap.add_argument("--label", default = None)
    ap.add_argument("--json-out", default = None)
    a = ap.parse_args()

    ref_m = json.load(open(os.path.join(a.ref_dir, "manifest.json")))
    cand_m = json.load(open(os.path.join(a.cand_dir, "manifest.json")))
    label = a.label or os.path.basename(a.cand_dir.rstrip("/"))

    print("reference: %s" % ref_m.get("model"))
    print("candidate: %s  (%s)" % (cand_m.get("model"), label))
    print()
    print("%-16s %6s  %10s %10s %10s   %7s %7s"
          % ("prompt", "tok", "KL mean", "KL median", "KL p99", "top-1", "top-5"))

    all_kl, tot_agree, tot_top5, tot_tok = [], 0, 0, 0
    result = {"label": label, "prompts": {}}
    for name, rp in ref_m["prompts"].items():
        if name not in cand_m["prompts"]:
            print("%-16s  MISSING in candidate" % name); continue
        cp = cand_m["prompts"][name]
        if rp["ids"] != cp["ids"]:
            # Not comparable: different tokenisation means position i is a
            # different prediction problem on each side.
            n = sum(1 for x, y in zip(rp["ids"], cp["ids"]) if x != y)
            print("%-16s  TOKENISATION MISMATCH (%d ids differ, %d vs %d tokens)"
                  % (name, n, len(rp["ids"]), len(cp["ids"])))
            sys.exit(3)
        T, V = rp["tokens"], rp["vocab"]
        kl, agree, top5 = score_prompt(
            os.path.join(a.ref_dir, rp["file"]),
            os.path.join(a.cand_dir, cp["file"]), T, V)
        all_kl.append(kl); tot_agree += agree; tot_top5 += top5; tot_tok += T
        print("%-16s %6d  %10.3e %10.3e %10.3e   %6.2f%% %6.2f%%"
              % (name, T, kl.mean(), np.median(kl), np.percentile(kl, 99),
                 100 * agree / T, 100 * top5 / T))
        result["prompts"][name] = {
            "tokens": T, "kl_mean": float(kl.mean()),
            "kl_median": float(np.median(kl)), "kl_p99": float(np.percentile(kl, 99)),
            "top1": agree / T, "top5": top5 / T}

    kl = np.concatenate(all_kl)
    print()
    print("%-16s %6d  %10.3e %10.3e %10.3e   %6.2f%% %6.2f%%"
          % ("ALL", tot_tok, kl.mean(), np.median(kl), np.percentile(kl, 99),
             100 * tot_agree / tot_tok, 100 * tot_top5 / tot_tok))
    result["total"] = {"tokens": tot_tok, "kl_mean": float(kl.mean()),
                       "kl_median": float(np.median(kl)),
                       "kl_p99": float(np.percentile(kl, 99)),
                       "kl_max": float(kl.max()),
                       "top1": tot_agree / tot_tok, "top5": tot_top5 / tot_tok}
    if a.json_out:
        json.dump(result, open(a.json_out, "w"), indent = 1)
        print("\nwrote %s" % a.json_out)


if __name__ == "__main__":
    main()
