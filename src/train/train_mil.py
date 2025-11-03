#!/usr/bin/env python3
import argparse, json, sys, math, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tokenization.tokenizer import tokenize, tokens_to_ids
    from models.nebula_model import NebulaTiny
except Exception:
    CUR = Path(__file__).resolve().parents[1]
    PARENT = CUR.parent / "src"
    if str(PARENT) not in sys.path:
        sys.path.insert(0, str(PARENT))
    from tokenization.tokenizer import tokenize, tokens_to_ids  # type: ignore
    from models.nebula_model import NebulaTiny  # type: ignore

def normalize_splits_dict(raw):
    if isinstance(raw, dict) and "splits" in raw and isinstance(raw["splits"], dict):
        return raw["splits"]
    return raw

def windows_by_events(text, events_per_window=16, stride_events=4):
    toks = text.split()
    idx = [i for i,t in enumerate(toks) if t.startswith("api:")]
    if not idx:
        return windows_by_tokens(text, events_per_window*8, max(1, stride_events)*8)
    starts = idx + [len(toks)]
    events = [" ".join(toks[starts[i]:starts[i+1]]) for i in range(len(starts)-1)]
    wins = []
    for i in range(0, len(events), stride_events):
        seg = events[i:i+events_per_window]
        if seg: wins.append(" ".join(seg))
    return wins

def windows_by_tokens(text, tokens_per_window=256, stride_tokens=64):
    toks = text.split()
    wins = []
    for i in range(0, len(toks), stride_tokens):
        seg = toks[i:i+tokens_per_window]
        if seg: wins.append(" ".join(seg))
    return wins

class SampleWindowsDataset(Dataset):
    """
    Returns (batch of window id-lists, label, lengths) for each SAMPLE.
    We'll pad to max_windows within the batch.
    """
    def __init__(self, data_file, splits, split_name, vocab, max_len,
                 unit="event", e_win=16, e_stride=4, t_win=256, t_stride=64, max_windows=64):
        self.vocab = vocab
        self.max_len = max_len
        self.unit = unit
        self.e_win, self.e_stride = e_win, e_stride
        self.t_win, self.t_stride = t_win, t_stride
        self.max_windows = max_windows

        id_set = set(splits[split_name])
        self.samples = []  # (label, [ids for each window])
        rows = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t", 2)
                if len(p) != 3: continue
                sid, y, txt = int(p[0]), int(p[1]), p[2]
                if sid not in id_set: continue
                rows.append((sid, y, txt))

        for _, y, txt in rows:
            wins = (windows_by_events(txt, e_win, e_stride) if unit=="event"
                    else windows_by_tokens(txt, t_win, t_stride))
            ids_list = [tokens_to_ids(tokenize(w), vocab, max_len=max_len) for w in wins]
            # clip to avoid huge memory
            if len(ids_list) > self.max_windows:
                ids_list = ids_list[:self.max_windows]
            self.samples.append((y, ids_list))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        y, ids_list = self.samples[idx]
        # pack windows in a single tensor [n_win, L]
        x = torch.tensor(ids_list, dtype=torch.long) if ids_list else torch.zeros(1, self.max_len, dtype=torch.long)
        lengths = torch.tensor([len(ids_list)], dtype=torch.long)
        return x, torch.tensor(y, dtype=torch.long), lengths

def collate_batch(batch):
    # batch: list of (x[nw,L], y, lengths)
    ys = torch.stack([b[1] for b in batch], dim=0)          # [B]
    lengths = [b[0].shape[0] for b in batch]
    max_w = max(lengths)
    L = batch[0][0].shape[-1]
    xs = torch.zeros(len(batch), max_w, L, dtype=torch.long)
    for i,(x,_,_) in enumerate(batch):
        xs[i,:x.shape[0],:] = x
    return xs, ys, torch.tensor(lengths, dtype=torch.long)  # xs: [B, max_w, L]

def load_ckpt(ckpt_path, vocab_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck["config"]
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    model = NebulaTiny(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], nhead=cfg["heads"],
                       num_layers=cfg["layers"], dim_feedforward=cfg["ff"], max_len=cfg["max_len"],
                       num_classes=2, chunk_size=cfg.get("chunk_size", 0))
    state = ck.get("model", ck.get("model_state"))
    model.load_state_dict(state)
    return model, vocab, cfg

def forward_windows(model, x):  # x: [B, W, L]
    B, W, L = x.shape
    x2 = x.view(B*W, L)
    logits = model(x2)  # [B*W, 2]
    logits = logits.view(B, W, 2)
    return logits

def evaluate_mil(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, loss_sum, tp, fp, tn, fn = 0, 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for xs, ys, lengths in loader:
            xs = xs.to(device); ys = ys.to(device)
            logits_w = forward_windows(model, xs)  # [B,W,2]
            # MIL max pooling over windows
            logits_max, _ = logits_w.max(dim=1)    # [B,2]
            loss = ce(logits_max, ys)
            loss_sum += loss.item()*xs.size(0); total += xs.size(0)
            pred = logits_max.argmax(-1)
            tp += int(((pred==1)&(ys==1)).sum())
            fp += int(((pred==1)&(ys==0)).sum())
            tn += int(((pred==0)&(ys==0)).sum())
            fn += int(((pred==0)&(ys==1)).sum())
    prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    return loss_sum/max(1,total), prec, rec, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--splits", default="data/manifests/splits.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--unit", choices=["event","token"], default="event")
    ap.add_argument("--events_per_window", type=int, default=16)
    ap.add_argument("--stride_events", type=int, default=4)
    ap.add_argument("--tokens_per_window", type=int, default=256)
    ap.add_argument("--stride_tokens", type=int, default=64)
    ap.add_argument("--max_windows", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)   # batch of samples
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--freeze_embed", action="store_true")
    args = ap.parse_args()

    model, vocab, cfg = load_ckpt(args.ckpt, args.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.freeze_embed:
        for p in model.tok_embed.parameters():
            p.requires_grad = False

    raw_splits = json.load(open(args.splits, "r", encoding="utf-8"))
    splits = normalize_splits_dict(raw_splits)
    train_ds = SampleWindowsDataset(args.data_file, splits, "train", vocab, cfg["max_len"],
                                    unit=args.unit, e_win=args.events_per_window, e_stride=args.stride_events,
                                    t_win=args.tokens_per_window, t_stride=args.stride_tokens,
                                    max_windows=args.max_windows)
    val_ds   = SampleWindowsDataset(args.data_file, splits, "val", vocab, cfg["max_len"],
                                    unit=args.unit, e_win=args.events_per_window, e_stride=args.stride_events,
                                    t_win=args.tokens_per_window, t_stride=args.stride_tokens,
                                    max_windows=args.max_windows)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_f1 = -1.0
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        model.train()
        tot, loss_sum = 0, 0.0
        for xs, ys, lengths in train_dl:
            xs = xs.to(device); ys = ys.to(device)
            opt.zero_grad()
            logits_w = forward_windows(model, xs)   # [B,W,2]
            logits_max, _ = logits_w.max(dim=1)     # [B,2]  (MIL: any-window)
            loss = ce(logits_max, ys)
            loss.backward()
            opt.step()
            loss_sum += loss.item()*xs.size(0); tot += xs.size(0)

        val_loss, prec, rec, f1 = evaluate_mil(model, val_dl, device)
        print(f"Epoch {epoch} | train_loss={loss_sum/max(1,tot):.4f} | val_loss={val_loss:.4f} | val_P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        torch.save({"config": cfg, "model": model.state_dict()}, out_dir/"last_mil.pt")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"config": cfg, "model": model.state_dict()}, out_dir/"best_mil.pt")
            print("  ↳ saved best_mil.pt")

if __name__ == "__main__":
    main()
