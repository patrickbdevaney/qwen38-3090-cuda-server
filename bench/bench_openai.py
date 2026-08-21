#!/usr/bin/env python3
"""
Benchmark harness for any OpenAI-compatible server.

Used identically for llama.cpp, vLLM and cuda_server so that every number in
reports/BENCHMARKS.md comes from the same prompts, the same sampling params and
the same timing code.  Per the directive: 3 warmup runs, 10 measured, report
median and p95.  Never reports a number it did not measure.

Usage:
  python3 bench/bench_openai.py --base-url http://127.0.0.1:8080/v1 \
      --model MODEL --suite bench/prompt_suite.json --tag llamacpp-ar
"""
import argparse, json, statistics, sys, time, urllib.request, urllib.error

def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer local-no-auth"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def stream(url, body, timeout=1800):
    """Returns (ttft_s, total_s, n_chunks, text). Times first content byte."""
    body = dict(body); body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer local-no-auth"})
    t0 = time.perf_counter(); ttft = None; n = 0; buf = []; usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: j = json.loads(payload)
            except json.JSONDecodeError: continue
            if j.get("usage"): usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {}) or {}
                piece = d.get("content") or d.get("reasoning_content") or ""
                if piece:
                    if ttft is None: ttft = time.perf_counter() - t0
                    n += 1; buf.append(piece)
    return ttft, time.perf_counter() - t0, n, "".join(buf), usage

def run_case(url, model, case, max_tokens, temperature, streaming):
    body = {"model": model,
            "messages": [{"role": "user", "content": case["prompt"]}],
            "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 1.0 if temperature == 0 else 0.95, "seed": 0}
    if streaming:
        ttft, total, n, text, usage = stream(url + "/chat/completions", body)
        ct = (usage or {}).get("completion_tokens") or n
        pt = (usage or {}).get("prompt_tokens")
    else:
        t0 = time.perf_counter()
        r = post(url + "/chat/completions", body)
        total = time.perf_counter() - t0; ttft = None
        text = r["choices"][0]["message"].get("content") or ""
        u = r.get("usage", {}); ct = u.get("completion_tokens"); pt = u.get("prompt_tokens")
        # llama.cpp-shaped timings, if present
        tm = r.get("timings") or {}
        if tm.get("predicted_per_second"):
            return {"decode_tok_s": tm["predicted_per_second"],
                    "prefill_tok_s": tm.get("prompt_per_second"),
                    "completion_tokens": tm.get("predicted_n", ct),
                    "prompt_tokens": pt, "ttft_s": ttft, "total_s": total,
                    "text": text}
    decode_s = (total - ttft) if (ttft is not None and total > ttft) else total
    return {"decode_tok_s": (ct - 1)/decode_s if ct and ct > 1 and decode_s > 0 else None,
            "prefill_tok_s": (pt/ttft) if (pt and ttft) else None,
            "completion_tokens": ct, "prompt_tokens": pt,
            "ttft_s": ttft, "total_s": total, "text": text}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--only", help="comma-separated case names")
    ap.add_argument("--out", help="write JSON results here")
    a = ap.parse_args()

    suite = json.load(open(a.suite))
    cases = suite["cases"]
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["name"] in keep]

    url = a.base_url.rstrip("/")
    results = {"tag": a.tag, "model": a.model, "max_tokens": a.max_tokens,
               "temperature": a.temperature, "warmup": a.warmup, "reps": a.reps,
               "cases": {}}
    print(f"=== {a.tag} :: {a.model} ===")
    print(f"{'case':22s} {'decode tok/s':>14s} {'p95':>8s} {'prefill tok/s':>14s} "
          f"{'TTFT s':>8s} {'ctok':>6s}")
    for c in cases:
        for _ in range(a.warmup):
            try: run_case(url, a.model, c, a.max_tokens, a.temperature, not a.no_stream)
            except Exception as e: print(f"  warmup error on {c['name']}: {e}"); break
        rows = []
        for _ in range(a.reps):
            try: rows.append(run_case(url, a.model, c, a.max_tokens, a.temperature, not a.no_stream))
            except Exception as e:
                print(f"  ERROR {c['name']}: {e}"); break
        if not rows: continue
        d = [r["decode_tok_s"] for r in rows if r["decode_tok_s"]]
        p = [r["prefill_tok_s"] for r in rows if r["prefill_tok_s"]]
        t = [r["ttft_s"] for r in rows if r["ttft_s"]]
        med = statistics.median(d) if d else float("nan")
        p95 = sorted(d)[int(len(d)*0.95)-1] if len(d) > 1 else med
        results["cases"][c["name"]] = {
            "decode_tok_s_median": med, "decode_tok_s_p95": p95,
            "decode_tok_s_all": d,
            "prefill_tok_s_median": statistics.median(p) if p else None,
            "ttft_s_median": statistics.median(t) if t else None,
            "completion_tokens": rows[0]["completion_tokens"],
            "prompt_tokens": rows[0]["prompt_tokens"],
            "sample_text": rows[0]["text"][:400]}
        r = results["cases"][c["name"]]
        print(f"{c['name']:22s} {med:14.2f} {p95:8.2f} "
              f"{(r['prefill_tok_s_median'] or 0):14.1f} "
              f"{(r['ttft_s_median'] or 0):8.3f} {r['completion_tokens'] or 0:6d}")
    if d:
        allmed = [v["decode_tok_s_median"] for v in results["cases"].values()]
        results["overall_decode_tok_s_median"] = statistics.median(allmed)
        print(f"{'OVERALL':22s} {statistics.median(allmed):14.2f}")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")

if __name__ == "__main__":
    sys.exit(main())
