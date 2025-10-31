#!/usr/bin/env python3
"""
Runs batch attacks and record metrics per-sample.
Supports simple insertion attack (sampling APIs from candidates) and uses model scoring.
Outputs JSON with per-sample orig_score, adv_score, success flag.
"""
import argparse, json, random, time
from pathlib import Path
import torch
from tqdm import tqdm

from tokenization.tokenizer import tokenize, tokens_to_ids
from models.nebula_model import NebulaTiny
from attacks.simple_attacks import insert_benign_events_text, replace_api_tokens

def load_model(ckpt, vocab_path):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ck["config"]
    vocab = json.load(open(vocab_path))
    model = NebulaTiny(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], nhead=cfg["heads"],
                       num_layers=cfg["layers"], dim_feedforward=cfg["ff"], max_len=cfg["max_len"],
                       num_classes=2, chunk_size=cfg.get("chunk_size", 0))
    state = ck.get("model", ck.get("model_state"))
    model.load_state_dict(state)
    model.eval()
    return model, vocab, cfg

def score_text(model, vocab, text, max_len):
    ids = tokens_to_ids(tokenize(text), vocab, max_len=max_len)
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
        prob_mal = torch.softmax(logits, dim=-1)[0,1].item()
    return float(prob_mal)

def run(args):
    model, vocab, cfg = load_model(args.ckpt, args.vocab)

    # API pool
    api_pool = json.load(open(args.api_candidates)) if args.api_candidates and Path(args.api_candidates).exists() else \
               ["api:ReadFile","api:CreateFileW","api:WriteFile","api:RegOpenKeyExW","api:connect","api:send","api:recv","api:Sleep"]

    # load dataset
    rows = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3: continue
            i, y, t = parts
            rows.append((int(i), int(y), t))
    random.shuffle(rows)
    if args.sample_limit:
        rows = rows[:args.sample_limit]

    # attack + score
    results = []
    attacked_count = 0
    for i, y, t in rows:
        if y != 1:  # attack only malicious
            continue
        if args.strategy == "insert":
            adv = insert_benign_events_text(t, api_pool, n=args.insert_n)
        else:
            adv = replace_api_tokens(t, api_pool, n_replace=max(5, args.insert_n // 10))
        orig_score = score_text(model, vocab, t, cfg["max_len"])
        adv_score  = score_text(model, vocab, adv, cfg["max_len"])
        success = adv_score < 0.5
        results.append({
            "id": i,
            "label": 1,
            "orig_score": orig_score,
            "adv_score": adv_score,
            "success": success
        })
        attacked_count += 1

    out = {
        "attack": f"{args.strategy}",
        "insert_n": args.insert_n,
        "sample_limit": args.sample_limit,
        "results": results
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"Saved {len(results)} attack results to {args.out} (attacked={attacked_count})")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--api_candidates", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--strategy", choices=["insert", "replace"], default="insert")
    p.add_argument("--insert_n", type=int, default=100)
    p.add_argument("--sample_limit", type=int, default=None)
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()