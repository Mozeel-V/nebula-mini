#!/usr/bin/env python3
"""
Quick benchmark: measures average time to tokenize+score windows for N samples,
using your model checkpoint and tokenizer. Prints numbers you can multiply
to estimate the full run time.

Usage:
python3 scripts/benchmark_window_scoring.py \
  --dataset real_data/processed/dataset_balanced.tsv \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --n 50 \
  --window_unit event \
  --events_per_window 16 \
  --stride_events 4 \
  --batch_size 64
"""
import time, json, argparse, sys
from pathlib import Path
import numpy as np
import torch

# import your repo tokenizer + model
from tokenization.tokenizer import tokenize, tokens_to_ids
from models.nebula_model import NebulaTiny

def load_model(ckpt_path, vocab_path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck.get("config", {}) or {}
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    vocab_size = cfg.get("vocab_size", len(vocab))
    d_model = cfg.get("d_model", 128)
    heads = cfg.get("heads", cfg.get("nhead", 4))
    layers = cfg.get("layers", cfg.get("num_layers", 2))
    ff = cfg.get("ff", cfg.get("dim_feedforward", 256))
    max_len = cfg.get("max_len", 512)
    model = NebulaTiny(vocab_size=vocab_size, d_model=d_model, nhead=heads,
                       num_layers=layers, dim_feedforward=ff, max_len=max_len)
    state = ck.get("model", ck.get("model_state", ck.get("model_state_dict", ck)))
    model.load_state_dict(state)
    model.to(device).eval()
    return model, vocab, max_len

def generate_windows_from_flat(flat_trace, window_unit, events_per_window, stride_events):
    # simplistic: if unit=event and flat uses " ||| " separator for events
    if " ||| " in flat_trace:
        tokens = flat_trace.split(" ||| ")
    else:
        tokens = flat_trace.split()
    windows = []
    if window_unit == "event":
        T = len(tokens)
        i = 0
        while i < T:
            win = tokens[i:i+events_per_window]
            if not win:
                break
            windows.append(" ||| ".join(win))
            i += stride_events
    else:
        # fallback token-windowing (not typical)
        step = events_per_window
        for i in range(0, len(tokens), step):
            windows.append(" ".join(tokens[i:i+events_per_window]))
    return windows

def batch_tokenize(windows, vocab, max_len):
    # returns list of id lists
    all_ids = []
    for w in windows:
        ids = tokens_to_ids(tokenize(w), vocab, max_len=max_len)
        all_ids.append(ids)
    return all_ids

def pad_batch(ids_batch, pad_id=0):
    maxl = max(len(x) for x in ids_batch)
    out = np.zeros((len(ids_batch), maxl), dtype=np.int64)
    for i, row in enumerate(ids_batch):
        out[i, :len(row)] = row
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--n", type=int, default=50, help="samples to benchmark")
    ap.add_argument("--window_unit", default="event")
    ap.add_argument("--events_per_window", type=int, default=16)
    ap.add_argument("--stride_events", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model, vocab, max_len = load_model(args.ckpt, args.vocab, device)
    print("Model loaded. max_len:", max_len, "vocab_size:", len(vocab))

    # read first N samples
    dataset = []
    with open(args.dataset, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if not ln.strip():
                continue
            sid, lab, trace = ln.rstrip("\n").split("\t", 2)
            dataset.append((sid, int(lab), trace))
            if len(dataset) >= args.n:
                break
    print("Loaded", len(dataset), "samples for benchmark")

    total_windows = 0
    tokenize_time = 0.0
    score_time = 0.0
    windows_batch = []
    per_sample_windows = []

    t0 = time.time()
    for sid, lab, trace in dataset:
        w = generate_windows_from_flat(trace, args.window_unit, args.events_per_window, args.stride_events)
        per_sample_windows.append(len(w))
        total_windows += len(w)
        # tokenize now, accumulate batches
        t1 = time.time()
        ids = batch_tokenize(w, vocab, max_len)
        tokenize_time += time.time() - t1
        windows_batch.extend(ids)

        # when we have enough for a batch, score them
        while len(windows_batch) >= args.batch_size:
            batch_ids = windows_batch[:args.batch_size]
            windows_batch = windows_batch[args.batch_size:]
            arr = pad_batch(batch_ids)
            tensor = torch.tensor(arr, dtype=torch.long, device=device)
            t2 = time.time()
            with torch.no_grad():
                logits = model(tensor)
                _ = torch.softmax(logits, dim=-1)[:,1]
            score_time += time.time() - t2

    # score remaining
    if windows_batch:
        arr = pad_batch(windows_batch)
        tensor = torch.tensor(arr, dtype=torch.long, device=device)
        t2 = time.time()
        with torch.no_grad():
            logits = model(tensor)
            _ = torch.softmax(logits, dim=-1)[:,1]
        score_time += time.time() - t2

    total_time = time.time() - t0
    avg_windows_per_sample = np.mean(per_sample_windows) if per_sample_windows else 0
    avg_tokens_per_window = None

    print("=== BENCHMARK RESULTS ===")
    print("Samples measured:", len(dataset))
    print("Total windows:", total_windows)
    print("Avg windows per sample:", avg_windows_per_sample)
    print("Total tokenize time (s): %.4f" % tokenize_time)
    print("Total scoring time (s):  %.4f" % score_time)
    print("Total elapsed (s):       %.4f" % total_time)
    print()
    print("Per-window tokenize time (s): %.6f" % (tokenize_time / total_windows if total_windows else 0))
    print("Per-window scoring time (s):  %.6f" % (score_time / total_windows if total_windows else 0))
    print("Per-sample total (s):         %.6f" % (total_time / len(dataset) if dataset else 0))
    print()
    print("To estimate full run time:")
    print("  total_samples_in_file * (avg_windows_per_sample) * (per-window scoring time) = estimated scoring seconds")
    print("  add tokenize/io overhead and margin for CPU contention / other tasks.")

if __name__ == '__main__':
    main()
