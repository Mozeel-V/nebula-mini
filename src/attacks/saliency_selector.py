# src/attacks/saliency_selector.py
#!/usr/bin/env python3
"""
Compute token-level saliency using gradients wrt the malware logit.
This version uses model.forward_from_embeddings(emb) to ensure gradients reach `emb`.
Outputs a JSON mapping: { sample_id (int): [top token indices] }.
"""

import argparse, json
from pathlib import Path
import torch
import sys

try:
    from tokenization.tokenizer import tokenize, tokens_to_ids
    from models.nebula_model import NebulaTiny
except Exception:
    CUR = Path(__file__).resolve().parents[2] / "src"
    if str(CUR) not in sys.path:
        sys.path.insert(0, str(CUR))
    from tokenization.tokenizer import tokenize, tokens_to_ids
    from models.nebula_model import NebulaTiny

def load_model(ckpt, vocab_path):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ck["config"]
    vocab = json.load(open(vocab_path))
    model = NebulaTiny(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        nhead=cfg["heads"],
        num_layers=cfg["layers"],
        dim_feedforward=cfg["ff"],
        max_len=cfg["max_len"],
        num_classes=2,
        chunk_size=cfg.get("chunk_size", 0),
    )
    # support both {"model": ...} and {"model_state": ...}
    state = ck.get("model", ck.get("model_state"))
    model.load_state_dict(state)
    model.eval()
    return model, vocab, cfg

def token_saliency(model, vocab, text, max_len, target_class=1):
    """
    Compute saliency by backpropagating the target class logit w.r.t. embedding tensor.
    Returns indices sorted by descending saliency magnitude.
    """
    ids = tokens_to_ids(tokenize(text), vocab, max_len=max_len)
    x = torch.tensor([ids], dtype=torch.long)   # [1, T]
    # Build embeddings and enable grad
    emb = model.embed(x)                         # [1, T, D]
    emb = emb.clone().detach().requires_grad_(True)

    # forward from embeddings (user-provided method)
    logits = model.forward_from_embeddings(emb)  # [1, C]
    logit = logits[0, target_class]
    model.zero_grad(set_to_none=True)
    logit.backward()

    if emb.grad is None:
        raise RuntimeError("emb.grad is None after backward. Ensure model.forward_from_embeddings uses 'emb' as input and that grads are enabled.")

    # sum absolute gradients across embedding dim -> [T]
    sal = emb.grad.abs().sum(dim=-1)[0]  # [T]
    idx_sorted = torch.argsort(sal, descending=True).tolist()
    return idx_sorted

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=32, help="top-K positions to keep per sample")
    p.add_argument("--sample_limit", type=int, default=None)
    args = p.parse_args()

    model, vocab, cfg = load_model(args.ckpt, args.vocab)

    rows = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3: continue
            i, y, t = parts
            rows.append((int(i), int(y), t))

    if args.sample_limit:
        rows = rows[: args.sample_limit]

    out = {}
    for i, y, t in rows:
        if y != 1:  # focus on malicious samples
            continue
        idx_sorted = token_saliency(model, vocab, t, max_len=cfg["max_len"])
        out[i] = idx_sorted[: args.k]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote saliency positions to", args.out)

if __name__ == "__main__":
    main()
