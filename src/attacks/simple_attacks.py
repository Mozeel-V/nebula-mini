#!/usr/bin/env python3
import json, argparse, random
from pathlib import Path
import importlib, sys

# Try to import from installed package; if that fails, add local src/ to path
try:
    from tokenization.tokenizer import tokenize, tokens_to_ids
except Exception:
    # add local src/ to path so imports work when running script directly
    THIS = Path(__file__).resolve()
    PROJ_ROOT = THIS.parents[2]    # ../.. -> project root when file in src/attacks
    SRC_DIR = PROJ_ROOT / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    # now import the module as if installed
    from tokenization.tokenizer import tokenize, tokens_to_ids

def insert_benign_events_text(text, n=50):
    pad = " ".join([f"api:ReadFile path:C:\\\\Windows\\\\Temp\\\\pad{ i }.tmp" for i in range(n)])
    return text + " " + pad

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--strategy", choices=["insert","replace"], default="insert")
    p.add_argument("--insert_n", type=int, default=100)
    p.add_argument("--sample_limit", type=int, default=None, help="limit number of samples to attack (shuffle applied)")
    args = p.parse_args()

    rows = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            i, y, t = parts
            rows.append((int(i), int(y), t))

    out_lines = []
    for i,y,t in rows:
        if y==1:
            if args.strategy=="insert":
                adv = insert_benign_events_text(t, n=args.insert_n)
            else:
                adv = t.replace("malicious-c2", "example")
            out_lines.append(f"{i}\t{y}\t{adv}")

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print("Wrote", len(out_lines), "adversarial samples to", out_path)

if __name__ == "__main__":
    main()