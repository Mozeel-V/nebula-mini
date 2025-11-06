#!/usr/bin/env python3
import json, argparse

def load_json(p):
    j = json.load(open(p, "r", encoding="utf-8"))
    return j, j.get("samples", {})

def max_or_neg_inf(scores):
    return max(scores) if scores else float("-inf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_json", required=True)   # e.g., results/window_eval_plot/window_eval.json
    ap.add_argument("--attack_json", required=True) # e.g., results/window_eval_attack/window_eval.json
    ap.add_argument("--zero_fp_json", required=True) # e.g., results/window_eval_plot/zero_fp_threshold.json
    args = ap.parse_args()

    base, B = load_json(args.base_json)
    att,  A = load_json(args.attack_json)
    thr = json.load(open(args.zero_fp_json))["threshold"]

    # use intersection of IDs to avoid reindex problems
    ids = sorted(set(B.keys()) & set(A.keys()), key=int)
    if not ids:
        raise SystemExit("[ERROR] No overlapping sample IDs between base and attack JSONs.")

    # focus on malware only (label from BASE)
    mal_ids = [sid for sid in ids if int(B[sid]["label"]) == 1]
    if not mal_ids:
        raise SystemExit("[ERROR] No malware samples on the intersection of IDs.")

    def detected(sample):
        sc = sample["scores"] or []
        return bool(sc and max_or_neg_inf(sc) >= thr)

    before = sum(detected(B[sid]) for sid in mal_ids)
    after  = sum(detected(A[sid]) for sid in mal_ids)

    was_detected_and_now_not = sum(1 for sid in mal_ids if detected(B[sid]) and not detected(A[sid]))
    total_detected_before = max(1, before)
    asr = was_detected_and_now_not / total_detected_before

    print(f"Threshold = {thr:.6f}")
    print(f"Overlapping malware samples = {len(mal_ids)}")
    print(f"Malware detected before attack: {before}/{len(mal_ids)}")
    print(f"Malware detected after attack:  {after}/{len(mal_ids)}")
    print(f"Attack Success Rate (ASR): {asr*100:.2f}%")
    # optional: top offenders
    inc, dec = [], []
    for sid in mal_ids:
        bmax = max_or_neg_inf(B[sid]["scores"] or [])
        amax = max_or_neg_inf(A[sid]["scores"] or [])
        inc.append((sid, amax-bmax))
        dec.append((sid, bmax-amax))
    inc.sort(key=lambda x: x[1], reverse=True)
    dec.sort(key=lambda x: x[1], reverse=True)
    print("Top 5 increases in P(mal) after attack:", inc[:5])
    print("Top 5 decreases in P(mal) after attack:", dec[:5])

if __name__ == "__main__":
    main()
