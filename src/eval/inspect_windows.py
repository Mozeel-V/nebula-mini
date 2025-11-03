#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

# local imports (works whether installed as package or run from src/)
try:
    from tokenization.tokenizer import tokenize, tokens_to_ids
except Exception:
    CUR = Path(__file__).resolve().parents[2] / "src"
    if str(CUR) not in sys.path:
        sys.path.insert(0, str(CUR))
    from tokenization.tokenizer import tokenize, tokens_to_ids  # type: ignore

def windows_by_events(text, events_per_window=32, stride_events=None):
    toks = text.split()
    api_idx = [i for i,t in enumerate(toks) if t.startswith("api:")]
    if not api_idx:
        return windows_by_tokens(text, tokens_per_window=events_per_window*8, stride_tokens=(stride_events or events_per_window)*8)
    starts = api_idx + [len(toks)]
    events = [" ".join(toks[starts[i]:starts[i+1]]) for i in range(len(starts)-1)]
    stride_events = stride_events or events_per_window
    wins = []
    for i in range(0, len(events), stride_events):
        seg = events[i:i+events_per_window]
        if seg: wins.append(" ".join(seg))
    return wins

def windows_by_tokens(text, tokens_per_window=256, stride_tokens=None):
    toks = text.split()
    stride_tokens = stride_tokens or tokens_per_window
    wins = []
    for i in range(0, len(toks), stride_tokens):
        seg = toks[i:i+tokens_per_window]
        if seg: wins.append(" ".join(seg))
    return wins

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--vocab_file", required=True)
    ap.add_argument("--window_unit", choices=["event","token"], default="event")
    ap.add_argument("--events_per_window", type=int, default=32)
    ap.add_argument("--stride_events", type=int, default=None)
    ap.add_argument("--tokens_per_window", type=int, default=512)
    ap.add_argument("--stride_tokens", type=int, default=128)
    ap.add_argument("--max_len", type=int, default=512, help="model max_len for tokens_to_ids")
    ap.add_argument("--sample_limit", type=int, default=50)
    args = ap.parse_args()

    vocab = json.load(open(args.vocab_file, "r", encoding="utf-8"))
    PAD = vocab.get("<pad>", 0)

    rows = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t", 2)
            if len(p) != 3: continue
            sid, y, txt = int(p[0]), int(p[1]), p[2]
            rows.append((sid, y, txt))
    if args.sample_limit:
        rows = rows[:args.sample_limit]

    win_counts, tok_counts, pad_fracs = [], [], []
    for sid, y, txt in rows:
        if args.window_unit == "event":
            wins = windows_by_events(txt, args.events_per_window, args.stride_events)
        else:
            wins = windows_by_tokens(txt, args.tokens_per_window, args.stride_tokens)
        win_counts.append(len(wins))
        for w in wins:
            ids = tokens_to_ids(tokenize(w), vocab, max_len=args.max_len)
            n_tok = int((np.array(ids) != PAD).sum())
            tok_counts.append(n_tok)
            pad_fracs.append(1.0 - (n_tok / max(len(ids),1)))

    def s(arr):
        if not arr: return (0,0,0)
        a = np.array(arr, dtype=float)
        return (float(np.min(a)), float(np.median(a)), float(np.max(a)))

    print("Samples inspected:", len(rows))
    print("Windows per sample (min / median / max):", s(win_counts))
    print("Non-PAD tokens per window (min / median / max):", s(tok_counts))
    print("PAD fraction per window (min / median / max):", s(pad_fracs))

    if tok_counts and np.median(tok_counts) < 50:
        print("[WARN] Windows are very short (<50 non-PAD tokens on median). Increase window size or use token-based windows.")
    if pad_fracs and np.median(pad_fracs) > 0.6:
        print("[WARN] Most of the input is PAD. Check max_len or window sizes/stride.")

if __name__ == "__main__":
    main()
