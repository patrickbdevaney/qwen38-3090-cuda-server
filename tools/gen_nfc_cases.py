#!/usr/bin/env python3
"""Generate NFC test vectors so the C++ implementation can be checked without Python."""
import json, random, sys, unicodedata

def main(out):
    cases = []
    rng = random.Random(7)
    # 1. every codepoint that has a canonical decomposition, alone and in context
    for cp in range(0x110000):
        ch = chr(cp)
        d = unicodedata.decomposition(ch)
        if d and not d.startswith("<"):
            cases.append(ch)
            cases.append("a" + ch + "b")
    # 2. Hangul, all three shapes
    for s in range(0xAC00, 0xD7A4, 97): cases.append(chr(s))
    cases += ["가", "각", "각", "각가"]
    # 3. combining-mark orderings, including out-of-order runs that require
    #    canonical reordering, and blocked-composition cases
    base = "aeoun"
    marks = [0x0301,0x0300,0x0308,0x0327,0x0304,0x030A,0x0323,0x0331,0x0653,0x093C]
    for _ in range(4000):
        b = rng.choice(base)
        k = rng.randint(1, 4)
        cases.append(b + "".join(chr(rng.choice(marks)) for _ in range(k)))
    # 4. composition exclusions and singletons
    cases += ["\u0958","\u0959","\u095F","\u2126","\u212A","\u212B",
              "\u0344","\u0F73","\u0F75","\uFB1D","\uFB2A","\u1E9B",
              "\u1E9B\u0323","q\u0307\u0323","q\u0323\u0307"]
    # 5. random multi-script strings
    pools = [range(0x41,0x5B), range(0x4E00,0x4E80), range(0x3040,0x309F),
             range(0x0400,0x0460), range(0x0600,0x0660), range(0x0900,0x0980),
             range(0x1F600,0x1F640)]
    for _ in range(2000):
        p = rng.choice(pools)
        n = rng.randint(1, 12)
        s = "".join(chr(rng.choice(list(p))) for _ in range(n))
        if rng.random() < 0.5:
            s += chr(rng.choice(marks))
        cases.append(s)
    # 6. plain ascii fast path
    cases += ["hello world", "", " ", "\n\t", "def f(x): return x"]

    seen = set(); uniq = []
    for c in cases:
        if c not in seen:
            seen.add(c); uniq.append(c)
    with open(out, "w", encoding="utf-8") as f:
        for c in uniq:
            f.write(json.dumps({"in": c, "nfc": unicodedata.normalize("NFC", c)},
                               ensure_ascii=False) + "\n")
    print(f"{len(uniq)} NFC cases -> {out} (unicodedata {unicodedata.unidata_version})")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/nfc_cases.jsonl")
