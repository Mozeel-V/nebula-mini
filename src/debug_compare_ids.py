#!/usr/bin/env python3
import json, sys
from pathlib import Path

def load(p):
    j = json.load(open(p, "r", encoding="utf-8"))
    S = j.get("samples", {})
    return j, {sid: S[sid] for sid in S}

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/debug_compare_ids.py BASE_JSON ATTACK_JSON")
        sys.exit(2)
    base_p, att_p = sys.argv[1], sys.argv[2]
    base, B = load(base_p)
    att,  A = load(att_p)
    bid = set(B.keys())
    aid = set(A.keys())
    inter = bid & aid
    only_b = sorted(bid - aid, key=int)
    only_a = sorted(aid - bid, key=int)
    print(f"[INFO] base ids: {len(bid)}, attack ids: {len(aid)}, intersection: {len(inter)}")
    if only_b:
        print(f"[WARN] {len(only_b)} ids only in BASE (first 10): {only_b[:10]}")
    if only_a:
        print(f"[WARN] {len(only_a)} ids only in ATTACK (first 10): {only_a[:10]}")
    # check labels on intersection
    mism = []
    for sid in sorted(inter, key=int):
        lb = int(B[sid]["label"]); la = int(A[sid]["label"])
        if lb != la:
            mism.append((sid, lb, la))
    if mism:
        print(f"[WARN] Label mismatch for {len(mism)} ids (first 10): {mism[:10]}")
    else:
        print("[OK] Labels match on intersection.")
    # print quick stats (max scores) on intersection
    def smax(v): 
        s = v.get("scores") or []
        return max(s) if s else float("-inf")
    inc = dec = same = 0
    for sid in inter:
        bmax = smax(B[sid]); amax = smax(A[sid])
        if amax > bmax: inc += 1
        elif amax < bmax: dec += 1
        else: same += 1
    print(f"[INFO] On intersecting ids: increased={inc}, decreased={dec}, same={same}")

if __name__ == "__main__":
    main()
