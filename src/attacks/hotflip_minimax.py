#!/usr/bin/env python3
"""
Gradient-guided, window-preserving token replacement attack:
- Keeps sequence/windows unchanged (no insertions).
- Replaces up to --max_flips tokens per malware sample.
- Uses benign-leaning candidate tokens (from compute_token_stats.py).
- Chooses replacements that most DECREASE P(malware) using first-order (hotflip) approximation.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

# ---- local imports
try:
    from tokenization.tokenizer import tokenize, tokens_to_ids
    from models.nebula_model import NebulaTiny
except Exception:
    CUR = Path(__file__).resolve().parents[1]
    if str(CUR) not in sys.path:
        sys.path.insert(0, str(CUR))
    from tokenization.tokenizer import tokenize, tokens_to_ids  # type: ignore
    from models.nebula_model import NebulaTiny  # type: ignore


def load_model(ckpt, vocab_path):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ck["config"]
    state = ck.get("model", ck.get("model_state"))
    vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
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
    model.load_state_dict(state)
    model.eval()
    return model, vocab, cfg


def logits_to_pmal(logits: torch.Tensor, m_idx: int) -> torch.Tensor:
    lo = logits - logits.max(dim=-1, keepdim=True).values
    e = torch.exp(lo)
    p = e / e.sum(dim=-1, keepdim=True)
    return p[..., m_idx]


def detect_malware_index(model, vocab, rows, max_len, cap=60) -> int:
    import numpy as np
    def mean_for_col(col):
        mm, bb = [], []
        for sid, y, txt in rows[:cap]:
            ids = tokens_to_ids(tokenize(txt), vocab, max_len=max_len)
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                lo = model(x)
                pm = logits_to_pmal(lo, col)[0].item()
            (mm if y == 1 else bb).append(pm)
        if not mm or not bb:
            return -1e9
        return float(np.mean(mm) - np.mean(bb))
    d0 = mean_for_col(0)
    d1 = mean_for_col(1)
    return 0 if d0 >= d1 else 1


def load_benign_candidates(stats_json, top_k=300):
    j = json.load(open(stats_json, "r", encoding="utf-8"))
    toks = [x["tok"] for x in j["tokens"] if x["tok"].startswith(("api:", "args:"))]
    return toks[:top_k]


def make_leaf_embeddings(emb_layer: torch.nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
    """
    emb_leaf is a TRUE leaf with grad: clone().detach().requires_grad_(True)
    """
    with torch.no_grad():
        emb = emb_layer(ids)               # non-leaf
    emb_leaf = emb.clone().detach().requires_grad_(True)
    emb_leaf.retain_grad()
    return emb_leaf


def score_from_leaf_emb(model, emb_leaf, m_idx):
    logits = model.forward_from_embeddings(emb_leaf)  # uses your pos -> layers -> pool -> cls
    pm = logits_to_pmal(logits, m_idx)                # [B]
    return pm, logits


def get_grad_wrt_emb(model, emb_layer, ids, m_idx):
    """
    Build LEAF embeddings, forward, backward, and return grad wrt emb [seq, d].
    """
    model.zero_grad(set_to_none=True)
    emb_leaf = make_leaf_embeddings(emb_layer, ids)   # [B,L,D], leaf with grad
    pm, _ = score_from_leaf_emb(model, emb_leaf, m_idx)
    loss = pm.sum()
    loss.backward()
    grad = emb_leaf.grad.detach()[0]                  # [seq, d]
    return grad


def hotflip_once(grad_vec, vocab, emb_weight, toks_list, cand_tokens):
    """
    One hotflip step over all positions: pick (pos, candidate) with most negative dot(grad, delta_emb).
    Returns (pos, token) or None.
    """
    tok2id = vocab
    W = emb_weight  # [V, d]
    positions = [i for i, t in enumerate(toks_list) if t.startswith("api:") or t.startswith("args:")]
    if not positions:
        return None

    best = None  # (dot, pos, cand)
    for i in positions:
        old_tok = toks_list[i]
        old_id = tok2id.get(old_tok, 0)
        g = grad_vec[i]  # [d]
        for cand in cand_tokens[:64]:
            if cand == old_tok:
                continue
            new_id = tok2id.get(cand, None)
            if new_id is None:
                continue
            delta = (W[new_id] - W[old_id])  # [d]
            dot = torch.dot(g, delta).item()
            if best is None or dot < best[0]:  # more negative = better (decrease P_mal)
                best = (dot, i, cand)
    return (best[1], best[2]) if best is not None else None


def hotflip_minimax(model, vocab, text, m_idx, cand_tokens, max_len, max_flips=3):
    """
    Full loop: recompute gradients after each committed flip.
    """
    emb_layer = model.embed
    emb_weight = emb_layer.weight.detach()

    toks = tokenize(text)
    ids = torch.tensor([tokens_to_ids(toks, vocab, max_len=max_len)], dtype=torch.long)

    # initial grad wrt embeddings
    grad = get_grad_wrt_emb(model, emb_layer, ids, m_idx)

    flips = 0
    for _ in range(max_flips):
        step = hotflip_once(grad, vocab, emb_weight, toks, cand_tokens)
        if step is None:
            break
        pos, new_tok = step
        old_tok = toks[pos]
        if new_tok == old_tok:
            break
        # commit flip
        toks[pos] = new_tok
        flips += 1
        # re-encode and recompute grad for next step
        ids = torch.tensor([tokens_to_ids(toks, vocab, max_len=max_len)], dtype=torch.long)
        grad = get_grad_wrt_emb(model, emb_layer, ids, m_idx)

    # final score
    with torch.no_grad():
        emb_leaf = make_leaf_embeddings(emb_layer, ids)
        pm, _ = score_from_leaf_emb(model, emb_leaf, m_idx)
        pm_val = pm.item()
    return " ".join(toks), pm_val, flips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--token_stats", required=True)
    ap.add_argument("--malware_index", default="auto")  # 0,1, or 'auto'
    ap.add_argument("--max_flips", type=int, default=3)
    args = ap.parse_args()

    # load rows
    rows = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t", 2)
            if len(p) != 3:
                continue
            sid, y, txt = int(p[0]), int(p[1]), p[2]
            rows.append((sid, y, txt))

    model, vocab, cfg = load_model(args.ckpt, args.vocab)
    m_idx = detect_malware_index(model, vocab, rows, cfg["max_len"]) if str(args.malware_index).lower()=="auto" else int(args.malware_index)
    print(f"[INFO] using malware_index={m_idx}")

    cand_tokens = load_benign_candidates(args.token_stats, top_k=300)
    if not cand_tokens:
        raise SystemExit("[ERROR] No benign candidate tokens found. Run compute_token_stats.py first.")

    out_lines = []
    changed = 0
    for sid, y, txt in rows:
        if y == 0:
            out_lines.append(f"{sid}\t{y}\t{txt}")
            continue
        new_txt, new_p, flips = hotflip_minimax(
            model, vocab, txt, m_idx, cand_tokens, cfg["max_len"], max_flips=args.max_flips
        )
        out_lines.append(f"{sid}\t{y}\t{new_txt}")
        changed += 1

    Path(args.out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"[INFO] wrote {args.out_file} (modified {changed} malware samples)")

if __name__ == "__main__":
    main()
