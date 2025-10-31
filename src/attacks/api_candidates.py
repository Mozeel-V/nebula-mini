#!/usr/bin/env python3
"""
Builds API candidate set from benign traces or a fallback hardcoded list.
Writes JSON list of candidate "event" text snippets that can be sampled by attacks.
"""
import argparse, json
from pathlib import Path
from collections import Counter

def extract_apis_from_raw(raw_dir, top_k=100):
    raw_dir = Path(raw_dir)
    cnt = Counter()
    for p in raw_dir.glob("*.json"):
        try:
            j = json.load(open(p))
        except Exception:
            continue
        for ev in j.get("events", []):
            api = ev.get("api")
            if api:
                cnt[api] += 1
    most = [api for api, _ in cnt.most_common(top_k)]
    return most

# hardcoded APIs as fallback
FALLBACK = [
    "ReadFile", "CreateFileW", "WriteFile", "RegOpenKeyExW", "RegSetValueExW", "CloseHandle",
    "CreateProcess", "GetProcAddress", "VirtualAllocEx", "WriteProcessMemory", "CreateRemoteThread",
    "connect", "send", "recv", "Sleep", "ListDirectory", "Stat", "QueryPerformanceCounter"
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", default="data/raw")
    p.add_argument("--out", default="checkpoints/api_candidates.json")
    p.add_argument("--top_k", type=int, default=50)
    args = p.parse_args()

    cand = extract_apis_from_raw(args.raw_dir, top_k=args.top_k)
    if not cand:
        cand = FALLBACK[:args.top_k]
    events = [f"api:{api}" for api in cand]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(events, f, indent=2)
    print("Wrote", len(events), "api candidates to", args.out)

if __name__ == "__main__":
    main()
