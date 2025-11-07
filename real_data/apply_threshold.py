#!/usr/bin/env python3
"""
Given a window_eval.json that contains per-sample 'scores' arrays, apply a fixed threshold
and produce a new window_eval_fixed.json which contains predicted classes and counts.
"""
import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window_eval", default="results/window_eval_plot/window_eval.json")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    we = json.load(open(args.window_eval, "r", encoding="utf-8"))
    samples = we.get("samples", {})
    fixed = {"threshold": args.threshold, "samples": {}, "summary": {}}
    tp = fp = tn = fn = 0
    for sid, info in samples.items():
        lab = int(info.get("label", 0))
        scores = info.get("scores", [])
        pred = 1 if (len(scores) > 0 and max(scores) >= args.threshold) else 0
        fixed["samples"][sid] = {
            "label": lab,
            "scores": scores,
            "pred_class": pred
        }
        if lab == 1 and pred == 1:
            tp += 1
        elif lab == 1 and pred == 0:
            fn += 1
        elif lab == 0 and pred == 1:
            fp += 1
        elif lab == 0 and pred == 0:
            tn += 1
    fixed["summary"] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    outpath = args.out or (Path(args.window_eval).with_name("window_eval_fixed.json"))
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(fixed, f, indent=2)
    print(f"Wrote fixed window eval to {outpath}, tp={tp}, fp={fp}, tn={tn}, fn={fn}")

if __name__ == "__main__":
    main()
