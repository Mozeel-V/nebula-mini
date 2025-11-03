#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

def windows_by_events(text, events_per_window=16, stride_events=4):
    toks = text.split()
    idx = [i for i,t in enumerate(toks) if t.startswith("api:")]
    if not idx:
        return [text]
    starts = idx + [len(toks)]
    events = [" ".join(toks[starts[i]:starts[i+1]]) for i in range(len(starts)-1)]
    wins = []
    for i in range(0, len(events), stride_events):
        seg = events[i:i+events_per_window]
        if seg: wins.append(" ".join(seg))
    return wins

def is_rule_malicious(window_text):
    w = window_text.lower()
    red_flags = [
        "api:virtualallocex", "api:writeprocessmemory", "api:createremotethread",
        "api:winexec", "api:createprocess", "api:shellExecute".lower(),
        "args:ip=<ip_public>", "api:connect", "api:recv", "api:send",
    ]
    return any(tok in w for tok in red_flags)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)      # dataset.txt (id \t label \t text)
    ap.add_argument("--out_file", required=True)       # windows.tsv (sid \t y_sample \t y_window \t text)
    ap.add_argument("--events_per_window", type=int, default=16)
    ap.add_argument("--stride_events", type=int, default=4)
    ap.add_argument("--max_per_sample", type=int, default=128)
    args = ap.parse_args()

    rows = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t", 2)
            if len(p)!=3: continue
            sid, y, txt = int(p[0]), int(p[1]), p[2]
            rows.append((sid, y, txt))

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    out = open(args.out_file, "w", encoding="utf-8")
    n = 0
    for sid, y, txt in rows:
        wins = windows_by_events(txt, args.events_per_window, args.stride_events)[:args.max_per_sample]
        for w in wins:
            yw = 1 if is_rule_malicious(w) else 0     # pseudo-label
            out.write(f"{sid}\t{y}\t{yw}\t{w}\n"); n += 1
    out.close()
    print("Wrote", n, "windows to", args.out_file)

if __name__ == "__main__":
    main()