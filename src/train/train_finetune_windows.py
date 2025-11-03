#!/usr/bin/env python3
import argparse, json, sys, math, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# local imports (works if run from repo root)
try:
    from tokenization.tokenizer import tokenize, tokens_to_ids
    from models.nebula_model import NebulaTiny
except Exception:
    CUR = Path(__file__).resolve().parents[1]  # src/
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

class WindowDataset(Dataset):
    def __init__(self, data_file, splits, split_name, vocab, max_len,
                 unit="event", e_win=16, e_stride=4, t_win=256, t_stride=64):
        self.vocab = vocab
        self.max_len = max_len
        self.samples = []   # (ids, label)
        id_set = set(splits[split_name])

        rows = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t", 2)
                if len(p) != 3: continue
                sid, y, txt = int(p[0]), int(p[1]), p[2]
                if sid not in id_set: continue
                rows.append((sid, y, txt))

        for sid, y, txt in rows:
            wins = (windows_by_events(txt, e_win, e_stride) if unit=="event"
                    else windows_by_tokens(txt, t_win, t_stride))
            for w in wins:
                ids = tokens_to_ids(tokenize(w), vocab, max_len=max_len)
                self.samples.append((ids, y))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        ids, y = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(y, dtype=torch.long)

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

def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, loss_sum, tp, fp, tn, fn = 0, 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(-1)
            tp += int(((pred==1)&(y==1)).sum())
            fp += int(((pred==1)&(y==0)).sum())
            tn += int(((pred==0)&(y==0)).sum())
            fn += int(((pred==0)&(y==1)).sum())
            total += x.size(0)
    prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    return loss_sum/total, prec, rec, f1

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
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--freeze_embed", action="store_true", help="freeze token embeddings")
    args = ap.parse_args()

    # load model + vocab
    model, vocab, cfg = load_ckpt(args.ckpt, args.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.freeze_embed:
        for p in model.tok_embed.parameters():
            p.requires_grad = False

    raw_splits = json.load(open(args.splits, "r", encoding="utf-8"))
    splits = normalize_splits_dict(raw_splits)

    train_ds = WindowDataset(args.data_file, splits, "train", vocab, cfg["max_len"],
                             unit=args.unit, e_win=args.events_per_window, e_stride=args.stride_events,
                             t_win=args.tokens_per_window, t_stride=args.stride_tokens)
    val_ds   = WindowDataset(args.data_file, splits, "val", vocab, cfg["max_len"],
                             unit=args.unit, e_win=args.events_per_window, e_stride=args.stride_events,
                             t_win=args.tokens_per_window, t_stride=args.stride_tokens)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    best_f1 = -1.0
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        model.train()
        tot, loss_sum = 0, 0.0
        for x, y in train_dl:
            x = x.to(device); y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()*x.size(0); tot += x.size(0)
        val_loss, prec, rec, f1 = evaluate(model, val_dl, device)
        print(f"Epoch {epoch} | train_loss={loss_sum/max(1,tot):.4f} | val_loss={val_loss:.4f} | val_P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        torch.save({"config": cfg, "model": model.state_dict()}, out_dir/"last_windows.pt")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"config": cfg, "model": model.state_dict()}, out_dir/"best_windows.pt")
            print("  ↳ saved best_windows.pt")

if __name__ == "__main__":
    main()
