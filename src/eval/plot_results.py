#!/usr/bin/env python3
"""
Plot simple evaluation results from attack JSON outputs.
Generates ASR bar chart and ROC-like comparison figure.
"""
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def load_results(path):
    j = json.load(open(path))
    res = [r for r in j.get("results", []) if r.get("success") is not None]
    return res

def compute_asr(res):
    if not res: return 0.0
    n = len(res)
    succ = sum(1 for r in res if r.get("success"))
    return succ / n

def plot_asr_bar(files, labels, out):
    asrs = [compute_asr(load_results(f)) for f in files]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, asrs)
    ax.set_ylabel("Attack Success Rate (ASR)")
    ax.set_ylim(0,1)
    for i,v in enumerate(asrs):
        ax.text(i, v+0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(out)
    print("Saved ASR bar to", out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True, help="attack result json files")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    files = args.inputs
    labels = args.labels
    plot_asr_bar(files, labels, args.out)

if __name__ == "__main__":
    main()
