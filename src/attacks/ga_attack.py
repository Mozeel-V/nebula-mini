#!/usr/bin/env python3
import argparse, random
from pathlib import Path
# minimal hill-climb mutation
def mutate_insert_padding(text, n=50):
    pad = " ".join([f"api:ReadFile path:C:\\\\Windows\\\\Temp\\\\pad{ i }.tmp" for i in range(n)])
    return text + " " + pad
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--n_pad", type=int, default=100)
    args = p.parse_args()
    rows = []
    with open(args.data_file) as f:
        for line in f:
            i,y,t = line.rstrip("\\n").split("\\t",2)
            rows.append((int(i), int(y), t))
    adv = []
    for i,y,t in rows:
        if y==1:
            best = t
            for it in range(args.iters):
                cand = mutate_insert_padding(best, n=args.n_pad)
                # here, without model scoring, keep the last candidate (placeholder)
                best = cand
            adv.append((i,y,best))
    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w") as f:
        f.write("\\n".join([f"{i}\\t{y}\\t{t}" for (i,y,t) in adv]))
    print("Wrote", len(adv), "adv traces to", args.out_file)

if __name__ == "__main__": 
    main()
