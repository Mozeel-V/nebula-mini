#!/usr/bin/env python3
"""
Compare before/after window_eval.json files using a fixed zero-FP threshold,
produce side-by-side per-sample PNGs and a CSV summary.

Usage:
python scripts/compare_before_after.py \
  --before results/window_eval_plot/window_eval.json \
  --after  results/window_eval_plot_after/window_eval.json \
  --zero_fp zero_fp_threshold.json \
  --out_dir results/comparisons \
  --only_malware
"""
import json, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def to_classes(scores, thr):
    return [1 if s >= thr - 1e-12 else 0 for s in scores]

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def plot_pair(before_cls, after_cls, sid, out_path, thr):
    n_before = len(before_cls)
    n_after = len(after_cls)
    x_before = np.arange(n_before)
    x_after = np.arange(n_after)

    fig, axes = plt.subplots(1,2, figsize=(10,3), sharey=True)
    axes[0].step(x_before, before_cls, where='mid', linewidth=1.8)
    axes[0].set_title(f"Before — sample {sid}")
    axes[0].set_xlabel("window index")
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_yticks([0,1])
    axes[0].set_yticklabels(["benign","malware"])

    axes[1].step(x_after, after_cls, where='mid', linewidth=1.8)
    axes[1].set_title(f"After — sample {sid}  (thr={thr:.6f})")
    axes[1].set_xlabel("window index")

    min_len = min(n_before, n_after)
    changed = []
    for i in range(min_len):
        if before_cls[i] == 1 and after_cls[i] == 0:
            changed.append(i)
    if changed:
        for ax in axes:
            for idx in changed:
                ax.axvspan(idx-0.3, idx+0.3, color='green', alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--zero_fp", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--only_malware", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    before = load_json(args.before)
    after = load_json(args.after)
    z = load_json(args.zero_fp)
    thr = float(z["threshold"])

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    plots_dir = out_dir / "per_sample"
    ensure_dir(plots_dir)

    sample_ids = set(before.get("samples", {}).keys()) | set(after.get("samples", {}).keys())
    sample_ids = sorted(sample_ids, key=lambda x: int(x) if str(x).isdigit() else x)

    summary_rows = []
    for sid in sample_ids:
        binfo = before["samples"].get(sid)
        ainfo = after["samples"].get(sid)
        if binfo is None or ainfo is None:
            continue
        label = int(binfo.get("label", 0))
        if args.only_malware and label != 1:
            continue

        b_scores = binfo.get("scores", [])
        a_scores = ainfo.get("scores", [])

        b_cls = to_classes(b_scores, thr)
        a_cls = to_classes(a_scores, thr)

        before_mal_windows = int(sum(1 for v in b_cls if v==1))
        after_mal_windows = int(sum(1 for v in a_cls if v==1))
        flipped = int(sum(1 for i in range(min(len(b_cls), len(a_cls))) if b_cls[i]==1 and a_cls[i]==0))

        out_png = plots_dir / f"sample_{sid}_before_after.png"
        if not out_png.exists() or args.overwrite:
            plot_pair(b_cls, a_cls, sid, out_png, thr)

        summary_rows.append({
            "sample": sid,
            "label": label,
            "n_windows_before": len(b_cls),
            "n_windows_after": len(a_cls),
            "mal_windows_before": before_mal_windows,
            "mal_windows_after": after_mal_windows,
            "flipped_1_to_0": flipped,
        })

    csv_path = out_dir / "comparison_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else
                                ["sample","label","n_windows_before","n_windows_after","mal_windows_before","mal_windows_after","flipped_1_to_0"])
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print(f"[DONE] Plots saved into {plots_dir}")
    print(f"[DONE] Summary CSV: {csv_path}")
    print(f"[INFO] Used fixed threshold from {args.zero_fp} -> thr={thr:.6f}")

if __name__ == "__main__":
    main()
