#!/usr/bin/env python3
"""
What it does:
1) Loads a window-eval JSON (format produced by src/eval/window_eval_plot.py).
   Expected structure: summary["samples"][sid]["scores"] -> list of P(mal) per window,
   and ["samples"][sid]["label"] -> 0/1.
2) Picks thr = max(max_per_benign) + eps  (if there are benign samples with scores).
   If there are no benign samples or benign samples have no windows, thr defaults to 1.0.
3) Computes TP/FP/TN/FN and prints a short report and detection rates.
4) Saves thr and the summary to out_json.
"""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

def safe_max(arr):
    return max(arr) if arr else float("-inf")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_json", required=True,
                   help="Path to window_eval.json produced by src/eval/window_eval.py")
    p.add_argument("--out_json", default=None,
                   help="Path to write zero-FP threshold summary (default: same folder /zero_fp_threshold.json)")
    p.add_argument("--eps", type=float, default=1e-6,
                   help="Small epsilon to add to benign max so threshold is strictly above any benign score.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    eval_path = Path(args.eval_json)
    if not eval_path.exists():
        print(f"[ERROR] eval_json not found: {eval_path}", file=sys.stderr)
        sys.exit(2)

    data = json.load(open(eval_path, "r", encoding="utf-8"))
    samples = data.get("samples", {})
    if not samples:
        print(f"[ERROR] No samples found in {eval_path}", file=sys.stderr)
        sys.exit(2)

    # collect per-sample maxima
    ben_maxes = []
    mal_maxes = []
    per_sample_max = {}
    labels_map = {}
    for sid, info in samples.items():
        label = int(info.get("label", 0))
        scores = info.get("scores", []) or []
        m = safe_max(scores) if scores else float("-inf")
        per_sample_max[sid] = m
        labels_map[sid] = label
        if label == 0:
            ben_maxes.append(m)
        else:
            mal_maxes.append(m)

    # compute zero-FP threshold
    if ben_maxes:
        benign_max_overall = max(ben_maxes)
        thr = float(benign_max_overall + args.eps)
    else:
        # no benign samples with scores -> set threshold to 1.0 to force zero-FP (no windows >= 1.0)
        thr = 1.0

    # compute confusion counts
    tp = fp = tn = fn = 0
    for sid, m in per_sample_max.items():
        y = labels_map[sid]
        detected = (m >= thr) if m != float("-inf") else False
        if detected and y == 1:
            tp += 1
        elif detected and y == 0:
            fp += 1
        elif (not detected) and y == 0:
            tn += 1
        elif (not detected) and y == 1:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    malware_detection_rate = tp / (tp + fn) if (tp + fn) else 0.0

    # print report
    print("==============================================")
    print("Zero-FP threshold computation")
    print("==============================================")
    print(f"Eval JSON: {eval_path}")
    print(f"Benign per-sample max (count): {len(ben_maxes)}")
    if ben_maxes:
        print(f"Benign max: min={min(ben_maxes):.6f}  med≈{sorted(ben_maxes)[len(ben_maxes)//2]:.6f}  max={max(ben_maxes):.6f}")
    print(f"Malware per-sample max (count): {len(mal_maxes)}")
    if mal_maxes:
        print(f"Malware max: min={min(mal_maxes):.6f}  med≈{sorted(mal_maxes)[len(mal_maxes)//2]:.6f}  max={max(mal_maxes):.6f}")
    print("----------------------------------------------")
    print(f"Chosen zero-FP threshold = {thr:.6f} (benign_max + eps={args.eps})")
    print("----------------------------------------------")
    print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}  (total={total})")
    print(f"Accuracy={accuracy:.4f}  Precision={precision:.4f}  Recall={recall:.4f}")
    print(f"Malware detection rate under zero-FP = {malware_detection_rate:.4f}")
    print("==============================================")

    # write out JSON summary
    out_json = Path(args.out_json) if args.out_json else eval_path.parent / "zero_fp_threshold.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "eval_json": str(eval_path),
        "threshold": thr,
        "eps": args.eps,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "malware_detection_rate": malware_detection_rate,
        "benign_count": len(ben_maxes), "malware_count": len(mal_maxes),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[INFO] Wrote summary to {out_json}")

if __name__ == "__main__":
    main()
