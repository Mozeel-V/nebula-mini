#!/usr/bin/env python3
"""
Saliency-guided adversarial attack:
- Compute saliency positions first (or provide via --saliency_json)
- For malicious samples, bias replacements/insertions around high-saliency token indices
- Evaluate orig vs adv scores and write JSON diagnostics
"""
import argparse, json, random
from pathlib import Path
import torch, sys

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
    model = NebulaTiny(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], nhead=cfg["heads"],
                       num_layers=cfg["layers"], dim_feedforward=cfg["ff"], max_len=cfg["max_len"],
                       num_classes=2, chunk_size=cfg.get("chunk_size", 0))
    model.load_state_dict(ck["model"] if "model" in ck else ck.get("model_state"))
    model.eval()
    return model, vocab, cfg

def score_text(model, vocab, text, max_len):
    ids = tokens_to_ids(tokenize(text), vocab, max_len=max_len)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        return float(torch.softmax(logits, dim=-1)[0,1].item())

def sample_event_from_api(api_token: str, idx: int = 0) -> str:
    api = api_token.split("api:")[-1]
    low = api.lower()
    if low in ("readfile", "createfilew", "writefile"):
        return f"api:{api} path:C:\\\\Windows\\\\Temp\\\\pad{idx}.tmp"
    if low in ("connect","send","recv"):
        return f"api:{api} ip:127.0.0.1"
    if low.startswith("reg"):
        return f"api:{api} path:HKEY_LOCAL_MACHINE\\\\Software\\\\Vendor"
    return f"api:{api}"

def saliency_biased_replace(text, important_idx, api_pool, n_replace=8):
    toks = text.split()
    api_positions = [i for i, t in enumerate(toks) if t.startswith("api:")]
    # intersect with important indices if possible
    targets = [i for i in important_idx if i in api_positions] or api_positions
    if not targets: return text
    for _ in range(min(n_replace, len(targets))):
        pos = random.choice(targets)
        toks[pos] = sample_event_from_api(random.choice(api_pool))
    return " ".join(toks)

def run(args):
    model, vocab, cfg = load_model(args.ckpt, args.vocab)
    api_pool = json.load(open(args.api_candidates))
    sal = json.load(open(args.saliency_json))  # {id: [pos...]}
    rows = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3: continue
            i, y, t = parts
            rows.append((int(i), int(y), t))
    out = []
    for i, y, t in rows[: args.sample_limit or len(rows)]:
        if y != 1: continue
        pos = sal.get(str(i)) or sal.get(i) or []
        orig = score_text(model, vocab, t, cfg["max_len"])
        adv_text = saliency_biased_replace(t, pos, api_pool, n_replace=args.n_replace)
        adv = score_text(model, vocab, adv_text, cfg["max_len"])
        success = adv < 0.5
        out.append({"id": i, "orig_score": orig, "adv_score": adv, "success": success})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"results": out}, f, indent=2)
    print("Wrote saliency attack results to", args.out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--api_candidates", required=True)
    p.add_argument("--saliency_json", required=True, help="from saliency_selector.py")
    p.add_argument("--out", required=True)
    p.add_argument("--n_replace", type=int, default=8)
    p.add_argument("--sample_limit", type=int, default=None)
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()
