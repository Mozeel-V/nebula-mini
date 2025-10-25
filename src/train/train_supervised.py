import argparse, os, json, time, numpy as np, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

CURR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(CURR)
if SRC not in sys.path: sys.path.append(SRC)

from tokenization.tokenizer import tokenize, tokens_to_ids
from models.nebula_model import NebulaTiny
from eval.metrics import compute_metrics

class TextDataset(Dataset):
    def __init__(self, data_file, split_indices, vocab, max_len=256):
        self.rows = []
        with open(data_file, "r") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 2)
                if len(parts) != 3:
                    continue  # skipping malformed/blank lines
                i, y, t = parts
                self.rows.append((int(i), int(y), t))
        self.id2row = {i: (y, t) for i, y, t in self.rows}
        self.indices = split_indices
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row_id = self.indices[idx]
        y, t = self.id2row[row_id]
        ids = tokens_to_ids(tokenize(t), self.vocab, self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(y, dtype=torch.long)

def train_epoch(model, loader, opt, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=-1)[:,1]
        ys.append(y.cpu().numpy())
        ps.append(prob.cpu().numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    return compute_metrics(ys, ps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--vocab_file", default="checkpoints/vocab.json")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--ff", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--chunk_size", type=int, default=0)
    args = ap.parse_args()

    with open(args.vocab_file, "r") as vf:
        vocab = json.load(vf)
    with open(args.splits, "r") as sf:
        splits = json.load(sf)["splits"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TextDataset(args.data_file, splits["train"], vocab, args.max_len)
    val_ds   = TextDataset(args.data_file, splits["val"], vocab, args.max_len)
    test_ds  = TextDataset(args.data_file, splits["test"], vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size)

    model = NebulaTiny(vocab_size=len(vocab), d_model=args.d_model, nhead=args.heads,
                       num_layers=args.layers, dim_feedforward=args.ff, max_len=args.max_len,
                       num_classes=2, chunk_size=args.chunk_size).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)

    os.makedirs("checkpoints", exist_ok=True)
    best_auc = 0.0
    for epoch in range(1, args.epochs+1):
        tr_loss = train_epoch(model, train_loader, opt, device)
        val_metrics = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch} | train_loss={tr_loss:.4f} | val_auc={val_metrics['auc']:.3f} "
              f"| val_f1={val_metrics['f1']:.3f} | TPR@1e-3={val_metrics['tpr_at_1e-3']:.3f}")
        
        save_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": opt.state_dict(),
            "config": {
                "vocab_size": len(vocab),
                "d_model": args.d_model,
                "heads": args.heads,
                "layers": args.layers,
                "ff": args.ff,
                "max_len": args.max_len,
                "chunk_size": args.chunk_size
            }
        }

        last_path = "checkpoints/last.pt"
        torch.save(save_dict, last_path)

        metrics_record = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_metrics": val_metrics,
            "timestamp": time.time()
        }
        with open("checkpoints/last_metrics.json", "w") as mf:
            json.dump(metrics_record, mf, indent=2)

        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save({"model": model.state_dict(),
                        "config": {"vocab_size": len(vocab), "d_model": args.d_model, "heads": args.heads,
                                   "layers": args.layers, "ff": args.ff, "max_len": args.max_len,
                                   "chunk_size": args.chunk_size}},
                       "checkpoints/best.pt")
            
            best_record = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_metrics": val_metrics,
                "timestamp": time.time()
            }
            with open("checkpoints/best_metrics.json", "w") as bf:
                json.dump(best_record, bf, indent=2)

    try:
        torch.save(model, "checkpoints/full_model.pkl")
    except Exception as e:
        print("Warning: saving full_model.pkl failed:", e)

    try:
        scripted = torch.jit.script(model.cpu())
        scripted.save("checkpoints/model_scripted.pt")
    except Exception as e:
        print("Warning: TorchScript save failed:", e)
        
    ckpt = torch.load("checkpoints/best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = eval_epoch(model, test_loader, device)
    print("TEST:", test_metrics)

    with open("checkpoints/test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

if __name__ == "__main__":
    main()
