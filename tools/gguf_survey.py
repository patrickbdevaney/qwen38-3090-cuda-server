#!/usr/bin/env python3
"""
Survey GGUF quants WITHOUT downloading them.

The tensor directory sits at the start of the file, so a range request for the
first few hundred KB is enough to read every tensor's type and shape. That gives
the exact BODY size -- what decode actually reads every token -- which is the
number that matters and the one the published file size hides, because the file
size also counts the embedding and the output head.
"""
import json, struct, sys, urllib.request

BPE = {  # bytes per element: block_bytes / block_elems
    0: 4, 1: 2, 30: 2,                     # F32, F16, BF16
    8: 34/32, 9: 36/32,                    # Q8_0, Q8_1
    2: 18/32, 3: 20/32, 6: 22/32, 7: 24/32,
    10: 84/256, 11: 110/256, 12: 144/256, 13: 176/256, 14: 210/256,
    16: 66/256, 17: 74/256, 18: 98/256, 19: 50/256, 20: 18/32,
    21: 110/256, 22: 82/256, 23: 136/256, 29: 56/256,
}
NAME = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
        10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",16:"IQ2_XXS",17:"IQ2_XS",
        18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",
        29:"IQ1_M",30:"BF16"}

class R:
    def __init__(self, b): self.b, self.p = b, 0
    def take(self, n):
        if self.p + n > len(self.b): raise EOFError("need more bytes")
        v = self.b[self.p:self.p+n]; self.p += n; return v
    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def s(self):
        n = self.u64(); return self.take(n).decode("utf-8", "replace")

def skip_value(r, t):
    if t == 8: r.s()
    elif t == 9:
        et = r.u32(); n = r.u64()
        for _ in range(n): skip_value(r, et)
    else:
        sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t]
        r.take(sz)

def parse(buf):
    r = R(buf)
    if r.u32() != 0x46554747: raise ValueError("bad magic")
    r.u32(); nt = r.u64(); nkv = r.u64()
    for _ in range(nkv):
        r.s(); skip_value(r, r.u32())
    out = []
    for _ in range(nt):
        name = r.s(); nd = r.u32()
        ne = [r.u64() for _ in range(nd)]
        ty = r.u32(); r.u64()
        out.append((name, ty, ne))
    return out

def fetch(repo, fn, nbytes):
    url = f"https://huggingface.co/{repo}/resolve/main/{fn}"
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes-1}"})
    return urllib.request.urlopen(req, timeout=180).read()

def survey(repo, fn):
    buf = b""
    for want in (12*1024*1024, 24*1024*1024, 48*1024*1024):
        buf = fetch(repo, fn, want)
        try:
            ts = parse(buf); break
        except EOFError:
            continue
    else:
        raise RuntimeError("header larger than 48 MB")
    body = emb = out = 0.0
    types = {}
    for name, ty, ne in ts:
        n = 1
        for d in ne: n *= d
        b = n * BPE.get(ty, 4)
        types[NAME.get(ty, str(ty))] = types.get(NAME.get(ty, str(ty)), 0) + b
        if "token_embd" in name: emb += b
        elif name.startswith("output."): out += b
        else: body += b
    G = 1 << 30
    top = sorted(types.items(), key=lambda kv: -kv[1])[:4]
    print(f"  {fn.replace('Qwen3.8-27B-','').replace('.gguf',''):<14} "
          f"body {body/G:7.3f}  head {out/G:6.3f}  emb {emb/G:6.3f}  "
          f"total {(body+out+emb)/G:7.3f}   " +
          " ".join(f"{k} {v/G:.2f}" for k, v in top))
    return body / G

if __name__ == "__main__":
    repo = "unsloth/Qwen3.8-27B-GGUF"
    files = sys.argv[1:] or [
        "Qwen3.8-27B-UD-Q4_K_XL.gguf", "Qwen3.8-27B-UD-IQ4_XS.gguf",
        "Qwen3.8-27B-UD-Q3_K_XL.gguf", "Qwen3.8-27B-UD-IQ3_S.gguf",
        "Qwen3.8-27B-UD-IQ3_XXS.gguf", "Qwen3.8-27B-UD-Q2_K_XL.gguf",
    ]
    print("all sizes GiB; BODY is what decode reads every token")
    for f in files:
        try: survey(repo, f)
        except Exception as e: print(f"  {f}: {e}")
