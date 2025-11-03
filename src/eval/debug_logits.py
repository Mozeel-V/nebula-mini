#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

try:
    from tokenization.tokenizer import tokenize, tokens_to_ids
    from models.nebula_model import NebulaTiny
except Exception:
    CUR = Path(__file__).resolve().parents[2] / "src"
    if str(CUR) not in sys.path:
        sys.path.insert(0, str(CUR))
    from tokenization.tokenizer import tokenize, tokens_to_ids  # type: ignore
    from models.nebula_model import NebulaTiny  # type: ignore

def windows_by_tokens(text, tokens_per_window=512, stride_tokens=128):
    toks = text.split()
    wins = []
    for i in range(0, len(toks), stride_tokens):
        seg = toks[i:i+tokens_per_window]
        if seg: wins.append(" ".join(seg))
    return wins

def load_model(ckpt, vocab_file):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ck["config"]
    vocab = json.load(open(vocab_file, "r", encoding="utf-8"))
    model = NebulaTiny(
        vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], nhead=cfg["heads"],
        num_layers=cfg["layers"], dim_feedforward=cfg["ff"], max_len=cfg["max_len"],
        num_classes=2, chunk_size=cfg.get("chunk_size", 0)
    )
    state = ck.get("model", ck.get("model_state"))
    model.load_state_dict(state)
    model.eval()
    return model, vocab, cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--sample_id", type=int, required=True)
    ap.add_argument("--tokens_per_window", type=int, default=512)
    ap.add_argument("--stride_tokens", type=int, default=128)
    args = ap.parse_args()

    model, vocab, cfg = load_model(args.ckpt, args.vocab)

    lines = open(args.data_file, "r", encoding="utf-8").read().splitlines()
    rec = None
    for line in lines:
        p = line.split("\t", 2)
        if len(p) == 3 and int(p[0]) == args.sample_id:
            rec = (int(p[0]), int(p[1]), p[2]); break
    if rec is None:
        print("Sample id not found"); return

    sid, y, txt = rec
    wins = windows_by_tokens(txt, args.tokens_per_window, args.stride_tokens)
    print(f"Sample {sid} label={y}, windows={len(wins)}")

    for idx, w in enumerate(wins[:10]):  # print first 10 windows to keep it readable
        ids = tokens_to_ids(tokenize(w), vocab, max_len=cfg["max_len"])
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)[0].cpu().numpy()  # shape [2]
        probs = np.exp(logits - np.logaddexp.reduce(logits))
        print(f"win {idx:02d}: logits={logits}, probs={probs}")

if __name__ == "__main__":
    main()
