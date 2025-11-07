#!/usr/bin/env python3
"""
Select the highest threshold that gives ZERO false positives in the given window_eval.json.

Input: results/window_eval_plot/window_eval.json
Output: zero_fp_threshold.json (contains thr, tp, fp, tn, fn, accuracy, precision, recall, malware_count, benign_count)
"""
import json
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict

def load_window_eval(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_counts(samples_dict, thr):
    tp = fp = tn = fn = 0
    for sid, info in samples_dict.items():
        label = int(info.get("label", 0))
        scores = info.get("scores", [])
        pred = 1 if (len(scores) > 0 and max(scores) >= thr) else 0
        if label == 1 and pred == 1:
            tp += 1
        elif label == 1 and pred == 0:
            fn += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
    return tp, fp, tn, fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window_eval", default="results/window_eval_plot/window_eval.json")
    ap.add_argument("--out", default="zero_fp_threshold.json")
    ap.add_argument("--grid", type=int, default=1000, help="number of candidate thresholds over [0,1]")
    args = ap.parse_args()

    we = load_window_eval(args.window_eval)
    samples = we.get("samples", {})
    # gather all per-window scores into a set for smarter threshold candidates
    all_scores = []
    for sid, info in samples.items():
        all_scores.extend(info.get("scores", []))
    if not all_scores:
        print("No scores found in window_eval.json -> cannot pick threshold.")
        return

    # candidate thresholds: unique sorted scores plus small margins
    uniq = sorted(set(all_scores))
    # add endpoints
    candidates = [0.0] + uniq + [1.0]
    # for robustness, sample an evenly spaced grid if too many unique scores
    if len(candidates) > args.grid:
        candidates = list(np.linspace(0.0, 1.0, args.grid))

    best = None  # tuple (tp, thr, counts)
    for thr in candidates:
        tp, fp, tn, fn = compute_counts(samples, thr)
        if fp == 0:
            if best is None or tp > best[0]:
                best = (tp, thr, (tp, fp, tn, fn))
    if best is None:
        # no threshold yields zero FP; fall back to threshold that minimizes FP then maximize TP
        best2 = None
        for thr in candidates:
            tp, fp, tn, fn = compute_counts(samples, thr)
            if best2 is None or (fp < best2[1]) or (fp == best2[1] and tp > best2[0]):
                best2 = (tp, fp, thr, (tp, fp, tn, fn))
        if best2:
            tp, fp, thr, counts = best2[0], best2[1], best2[2], best2[3]
            selected_thr = thr
            tp, fp, tn, fn = counts
        else:
            print("Could not find threshold.")
            return
    else:
        tp, selected_thr, (tp, fp, tn, fn) = best

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    out = {
        "eval_json": args.window_eval,
        "threshold": float(selected_thr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "malware_count": sum(1 for s in samples.values() if int(s.get("label",0))==1),
        "benign_count": sum(1 for s in samples.values() if int(s.get("label",0))==0)
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote zero-FP threshold -> {args.out} thr={selected_thr:.6f} tp={tp} fp={fp} tn={tn} fn={fn}")

if __name__ == "__main__":
    main()
