#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_results(path):
    d = json.load(open(path))
    res = d.get("results", [])
    attack = d.get("attack", Path(path).stem)
    meta = {k:v for k,v in d.items() if k!="results"}
    return attack, res, meta

def asr(results):
    if not results: return 0.0
    return 100.0 * (sum(1 for r in results if r.get("success")) / len(results))

def score_arrays(results):
    orig = np.array([r["orig_score"] for r in results], dtype=float)
    adv  = np.array([r["adv_score"]  for r in results], dtype=float)
    return orig, adv

def plot_hist(orig_adv_list, labels, out):
    plt.figure()
    for (orig, adv), lab in zip(orig_adv_list, labels):
        # plot hist of deltas (adv - orig)
        delta = adv - orig
        plt.hist(delta, bins=30, alpha=0.5, label=lab, density=True)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Score shift (adv - orig)")
    plt.ylabel("Density")
    plt.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1) / len(xs)
    return xs, ys

def plot_cdf(orig_adv_list, labels, out):
    plt.figure()
    for (orig, adv), lab in zip(orig_adv_list, labels):
        # Show CDF of adv scores
        xs, ys = ecdf(adv)
        plt.plot(xs, ys, label=f"{lab} (adv)")
    plt.xlabel("Malware probability")
    plt.ylabel("CDF")
    plt.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

def plot_ga_convergence(ga_meta_list, labels, out):
    plt.figure()
    plotted = False
    for meta, lab in zip(ga_meta_list, labels):
        conv = meta.get("convergence")
        if conv:
            plt.plot(range(1, len(conv)+1), conv, label=lab)
            plotted = True
    if not plotted:
        return
    plt.xlabel("Iteration")
    plt.ylabel("Best malware score (lower is better)")
    plt.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

def plot_asr_vs_insert(insert_metas, insert_asrs, out):
    if not insert_metas: return
    xs, ys = [], []
    for meta, a in zip(insert_metas, insert_asrs):
        n = meta.get("insert_n")
        if n is not None:
            xs.append(n); ys.append(a)
    if not xs: return
    order = np.argsort(xs)
    xs = np.array(xs)[order]; ys = np.array(ys)[order]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Insert_n (number of inserted events)")
    plt.ylabel("ASR (%)")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True,
                    help="List of JSON files: e.g. insert_results.json ga_results.json saliency_results.json")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--insert_sweep", nargs="*", default=[],
                    help="Optional: multiple INSERT result JSONs with different insert_n to build ASR vs insert curve")
    args = ap.parse_args()

    attacks, orig_adv_list, labels, metas, asrs = [], [], [], [], []
    ga_metas = []
    for path in args.results:
        att, res, meta = load_results(path)
        if not res: continue
        o, a = score_arrays(res)
        attacks.append(att)
        orig_adv_list.append((o, a))
        labels.append(att.capitalize())
        metas.append(meta)
        asrs.append(asr(res))
        if att.lower()=="ga":
            ga_metas.append(meta)

    # 1) score shift histogram
    plot_hist(orig_adv_list, labels, Path(args.out_dir)/"score_shift_hist.png")
    # 2) CDF of adv scores
    plot_cdf(orig_adv_list, labels, Path(args.out_dir)/"score_shift_cdf.png")
    # 3) GA convergence (if available)
    plot_ga_convergence(ga_metas, [l for l,a in zip(labels, attacks) if a.lower()=="ga"], Path(args.out_dir)/"ga_convergence.png")

    # 4) ASR vs insert_n (needs multiple INSERT JSONs with different insert_n)
    insert_metas, insert_asrs = [], []
    for p in (args.insert_sweep or []):
        att, res, meta = load_results(p)
        if att.lower()!="insert" or not res: continue
        insert_metas.append(meta); insert_asrs.append(asr(res))
    plot_asr_vs_insert(insert_metas, insert_asrs, Path(args.out_dir)/"asr_vs_insert.png")

    print("Wrote plots to", args.out_dir)

if __name__ == "__main__":
    main()
