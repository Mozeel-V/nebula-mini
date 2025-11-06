#!/usr/bin/env python3
"""
Compare baseline vs attacked datasets at the window level:
- counts how many windows per sample (did attack add more?)
- compares max P(mal) per sample (did attack increase the max window score?)
"""
import argparse, json
from statistics import mean

def max_or_neginf(v): return max(v) if v else float("-inf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_json", required=True)
    ap.add_argument("--attack_json", required=True)
    ap.add_argument("--thr_json", required=True)
    args = ap.parse_args()

    base = json.load(open(args.base_json, "r", encoding="utf-8"))
    att  = json.load(open(args.attack_json, "r", encoding="utf-8"))
    thr  = json.load(open(args.thr_json, "r", encoding="utf-8"))["threshold"]

    B, A = base["samples"], att["samples"]
    ids = sorted(set(B.keys()) & set(A.keys()), key=int)

    inc_cnt = dec_cnt = same_cnt = 0
    added_wins = more = less = same_w = 0

    deltas = []  # (sid, before_max, after_max, before_w, after_w)
    for sid in ids:
        b, a = B[sid], A[sid]
        bw, aw = b["n_windows"], a["n_windows"]
        if aw > bw: more += 1
        elif aw < bw: less += 1
        else: same_w += 1

        bmax = max_or_neginf(b["scores"])
        amax = max_or_neginf(a["scores"])
        if amax > bmax: inc_cnt += 1
        elif amax < bmax: dec_cnt += 1
        else: same_cnt += 1
        deltas.append((sid, bmax, amax, bw, aw))

    print(f"[IDs] overlap={len(ids)}")
    print(f"[Windows] attack added more windows in {more} / less in {less} / same in {same_w}")
    print(f"[Max P(mal)] increased for {inc_cnt}, decreased for {dec_cnt}, same for {same_cnt}")
    # show a few examples where windows increased a lot
    deltas.sort(key=lambda x: (x[2]-x[1]), reverse=True)
    print("Top 5 increases (sid, bmax, amax, bw, aw):")
    for row in deltas[:5]:
        print("  ", row)
    print("Top 5 decreases (sid, bmax, amax, bw, aw):")
    for row in sorted(deltas, key=lambda x: (x[2]-x[1]))[:5]:
        print("  ", row)

if __name__ == "__main__":
    main()
