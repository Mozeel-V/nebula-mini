#!/usr/bin/env python3
"""
Simple GA/hill-climb mutation attack that uses the model score as fitness.
Mutations sample APIs from candidates and insert/replace to reduce model malicious score.
"""
import argparse, random, json, copy
from pathlib import Path
import torch

from tokenization.tokenizer import tokenize, tokens_to_ids
from models.nebula_model import NebulaTiny

def load_model(ckpt, vocab_path):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ck["config"]
    vocab = json.load(open(vocab_path))
    model = NebulaTiny(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], nhead=cfg["heads"],
                       num_layers=cfg["layers"], dim_feedforward=cfg["ff"], max_len=cfg["max_len"])
    model.load_state_dict(ck["model"] if "model" in ck else ck.get("model_state"))
    model.eval()
    return model, vocab, cfg

def score_text(model, vocab, text, max_len):
    ids = tokens_to_ids(tokenize(text), vocab, max_len=max_len)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        return float(torch.softmax(logits, dim=-1)[0,1].item())

def sample_event_from_api(api_token, idx=0):
    api = api_token.split("api:")[-1]
    if api.lower() in ("readfile","createfilew","writefile"):
        return f"api:{api} path:C:\\\\Windows\\\\Temp\\\\pad{idx}.tmp"
    return f"api:{api}"

def mutate_insert(text, api_pool, n_insert=50):
    pads = [sample_event_from_api(random.choice(api_pool), i) for i in range(n_insert)]
    return text + " " + " ".join(pads)

def mutate_replace(text, api_pool, n_replace=10):
    toks = text.split()
    # find positions with api: prefix
    api_pos = [i for i,t in enumerate(toks) if t.startswith("api:")]
    if not api_pos:
        return text
    for _ in range(n_replace):
        pos = random.choice(api_pos)
        new_ev = sample_event_from_api(random.choice(api_pool))
        toks[pos] = new_ev
    return " ".join(toks)

def hill_climb(original_text, model, vocab, cfg, api_pool, iters=30):
    best = original_text
    best_score = score_text(model, vocab, best, cfg["max_len"])
    for it in range(iters):
        if random.random() < 0.5:
            cand = mutate_insert(best, api_pool, n_insert=20)
        else:
            cand = mutate_replace(best, api_pool, n_replace=5)
        sc = score_text(model, vocab, cand, cfg["max_len"])
        if sc < best_score:
            best = cand; best_score = sc
    return best, best_score

def run(args):
    model, vocab, cfg = load_model(args.ckpt, args.vocab)
    api_pool = json.load(open(args.api_candidates))
    rows = []
    with open(args.data_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t",2)
            if len(parts)!=3: continue
            i,y,t = parts
            rows.append((int(i), int(y), t))
    random.shuffle(rows)
    if args.sample_limit: rows = rows[:args.sample_limit]
    out = []
    for i,y,t in rows:
        if y==1:
            best, best_score = hill_climb(t, model, vocab, cfg, api_pool, iters=args.iters)
            orig = score_text(model, vocab, t, cfg["max_len"])
            success = best_score < 0.5
            out.append({"id": i, "orig_score": orig, "adv_score": best_score, "success": success, "adv_text": best})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w") as f:
        json.dump({"meta":{"n":len(out)}, "results": out}, f, indent=2)
    print("Wrote GA attack results to", args.out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--api_candidates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=25)
    p.add_argument("--sample_limit", type=int, default=None)
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()
