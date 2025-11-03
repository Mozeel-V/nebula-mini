#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
from collections import Counter
import numpy as np

try:
    from tokenization.tokenizer import tokenize
except Exception:
    CUR = Path(__file__).resolve().parents[2] / "src"
    if str(CUR) not in sys.path:
        sys.path.insert(0, str(CUR))
    from tokenization.tokenizer import tokenize  # type: ignore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--vocab_file", required=True)
    ap.add_argument("--sample_limit", type=int, default=500)
    args = ap.parse_args()

    vocab = json.load(open(args.vocab_file, "r", encoding="utf-8"))
    itos = {int(v): k for k, v in vocab.items()}
    stoi = vocab

    unk = stoi.get("<unk>", None)
    pad = stoi.get("<pad>", None)

    oov_rates = []
    token_freq = Counter()
    line_count = 0

    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3: continue
            _, _, text = parts
            toks = tokenize(text)
            line_count += 1
            if line_count > args.sample_limit: break
            n = len(toks)
            oov = 0
            for t in toks:
                token_freq[t] += 1
                if t not in stoi:
                    oov += 1
            oov_rates.append(oov / max(1, n))

    top = token_freq.most_common(25)
    print(f"Samples scanned: {line_count}")
    print(f"OOV rate: min/med/max = {np.min(oov_rates):.3f} / {np.median(oov_rates):.3f} / {np.max(oov_rates):.3f}")
    print("Top 25 tokens:", [t for t,_ in top])

if __name__ == "__main__":
    main()
