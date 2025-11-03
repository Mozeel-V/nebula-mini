#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from tokenization.tokenizer import tokenize, tokens_to_ids
    from models.nebula_model import NebulaTiny
except Exception:
    CUR = Path(__file__).resolve().parents[1]
    PARENT = CUR.parent / "src"
    if str(PARENT) not in sys.path: sys.path.insert(0, str(PARENT))
    from tokenization.tokenizer import tokenize, tokens_to_ids  # type: ignore
    from models.nebula_model import NebulaTiny  # type: ignore

class WinDS(Dataset):
    def __init__(self, tsv, vocab, max_len):
        self.vocab, self.max_len = vocab, max_len
        self.rows = []
        with open(tsv, "r", encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t", 3)
                if len(p)!=4: continue
                sid, y_s, y_w, txt = int(p[0]), int(p[1]), int(p[2]), p[3]
                self.rows.append((sid, y_s, y_w, txt))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        sid, y_s, y_w, txt = self.rows[i]
        ids = tokens_to_ids(tokenize(txt), self.vocab, max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(y_w, dtype=torch.long), sid

def load_ckpt(ckpt, vocab_path):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ck["config"]
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    model = NebulaTiny(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], nhead=cfg["heads"],
        num_layers=cfg["layers"], dim_feedforward=cfg["ff"], max_len=cfg["max_len"],
        num_classes=2, chunk_size=cfg.get("chunk_size", 0)
    )
    state = ck.get("model", ck.get("model_state"))
    model.load_state_dict(state)
    return model, vocab, cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_tsv", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out_ckpt", default="checkpoints/best_window_supervised.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    model, vocab, cfg = load_ckpt(args.ckpt, args.vocab)
    ds = WinDS(args.windows_tsv, vocab, cfg["max_len"])
    n = len(ds); split = int(0.9*n)
    train_ds, val_ds = torch.utils.data.random_split(ds, [split, n-split], generator=torch.Generator().manual_seed(42))
    tr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    va = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = -1
    for e in range(1, args.epochs+1):
        # train
        model.train(); tot=0; loss_sum=0
        for x,y,_ in tr:
            x,y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y); loss.backward(); opt.step()
            loss_sum += loss.item()*x.size(0); tot += x.size(0)
        # val
        model.eval(); tp=fp=tn=fn=0; vtot=0; vloss=0
        with torch.no_grad():
            for x,y,_ in va:
                x,y = x.to(dev), y.to(dev)
                logits = model(x); loss = ce(logits, y)
                pred = logits.argmax(-1)
                tp += int(((pred==1)&(y==1)).sum()); fp += int(((pred==1)&(y==0)).sum())
                tn += int(((pred==0)&(y==0)).sum()); fn += int(((pred==0)&(y==1)).sum())
                vloss += loss.item()*x.size(0); vtot += x.size(0)
        prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
        f1 = 2*prec*rec/(prec+rec+1e-12)
        print(f"Epoch {e} | train_loss={loss_sum/max(1,tot):.4f} | val_loss={vloss/max(1,vtot):.4f} | P={prec:.3f} R={rec:.3f} F1={f1:.3f}")
        torch.save({"config": cfg, "model": model.state_dict()}, args.out_ckpt)
        if f1>best_f1:
            best_f1=f1
            torch.save({"config": cfg, "model": model.state_dict()}, args.out_ckpt)
            print("  ↳ saved", args.out_ckpt)

if __name__ == "__main__":
    main()