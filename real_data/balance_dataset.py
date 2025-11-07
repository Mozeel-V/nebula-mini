#!/usr/bin/env python3
"""
Balance a TSV dataset into 50/50 (malware / benign) by undersampling the majority class.
Input format: id \t label \t trace
Output: data/processed/dataset_balanced.tsv
"""
import argparse
import random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/dataset_raw.tsv")
    ap.add_argument("--output", default="data/processed/dataset_balanced.tsv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    samples = {0: [], 1: []}
    with open(inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            sid, lab, trace = parts[0], parts[1], parts[2]
            lab = int(lab)
            samples[lab].append((sid, lab, trace))

    n0 = len(samples[0])
    n1 = len(samples[1])
    print(f"Counts before: benign={n0}, malware={n1}")

    if n0 == 0 or n1 == 0:
        print("Warning: one class is empty — nothing to balance.")
        chosen = samples[0] + samples[1]
    else:
        target = min(n0, n1)
        chosen0 = random.sample(samples[0], target) if n0 > target else samples[0]
        chosen1 = random.sample(samples[1], target) if n1 > target else samples[1]
        chosen = chosen0 + chosen1
    random.shuffle(chosen)

    with open(outp, "w", encoding="utf-8") as f:
        for sid, lab, trace in chosen:
            f.write(f"{sid}\t{lab}\t{trace}\n")

    print(f"Wrote balanced dataset with {len(chosen)} samples to {outp} (balanced 50/50).")

if __name__ == "__main__":
    main()
