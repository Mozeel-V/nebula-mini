#!/usr/bin/env python3
import argparse, json, math
from collections import Counter, defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)  # data/processed/dataset.txt
    ap.add_argument("--out_json", required=True)   # results/token_stats.json
    ap.add_argument("--min_freq", type=int, default=5)
    args = ap.parse_args()

    tok_mal = Counter()
    tok_ben = Counter()
    N_mal = N_ben = 0

    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t", 2)
            if len(p) != 3: continue
            _, y, txt = int(p[0]), int(p[1]), p[2]
            toks = txt.split()
            if y == 1:
                tok_mal.update(toks); N_mal += 1
            else:
                tok_ben.update(toks); N_ben += 1

    # PMI-ish score: log ((f_ben+1)/(N_ben+1)) - log((f_mal+1)/(N_mal+1))
    stats = []
    for t in set(tok_mal)|set(tok_ben):
        f_b = tok_ben[t]; f_m = tok_mal[t]
        if f_b + f_m < args.min_freq: continue
        s = math.log((f_b+1)/(N_ben+1)) - math.log((f_m+1)/(N_mal+1))
        stats.append((t, f_b, f_m, s))

    stats.sort(key=lambda x: x[3], reverse=True)  # most benign-leaning first
    out = {
        "N_ben": N_ben, "N_mal": N_mal, "min_freq": args.min_freq,
        "tokens": [{"tok":t, "f_ben":fb, "f_mal":fm, "score":s} for (t,fb,fm,s) in stats]
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] wrote {args.out_json} with {len(out['tokens'])} tokens (min_freq={args.min_freq})")

if __name__ == "__main__":
    main()
