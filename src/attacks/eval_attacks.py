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
        prob = float(torch.softmax(logits, dim=-1)[0,1].item())
    return prob

def sample_api_event(api_token):
    # basic mapping: api:ReadFile -> "api:ReadFile path:C:\\Windows\\Temp\\padX.tmp"
    api = api_token.split("api:")[-1]
    if api.lower() in ("readfile","createfilew","writefile"):
        return f"api:{api} path:C:\\\\Windows\\\\Temp\\\\pad.tmp"
    if api.lower() in ("connect","send","recv"):
        return f"api:{api} ip:127.0.0.1"
    if api.lower().startswith("reg"):
        return f"api:{api} path:HKEY_LOCAL_MACHINE\\\\Software\\\\Vendor"
    return f"api:{api}"

def make_inserted(text, api_pool, insert_n=50):
    pads = []
    for i in range(insert_n):
        tok = random.choice(api_pool)
        pads.append(sample_api_event(tok))
    return text + " " + " ".join(pads)

def run(args):
    model, vocab, cfg = load_model(args.ckpt, args.vocab)
    max_len = cfg["max_len"]
    api_pool = json.load(open(args.api_candidates))
    rows = []
    with open(args.data_file) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t",2)
            if len(parts)!=3: continue
            i,y,t = parts
            rows.append((int(i), int(y), t))
    random.shuffle(rows)
    if args.sample_limit:
        rows = rows[:args.sample_limit]

    results = []
    t0 = time.time()
    for i,y,t in tqdm(rows, total=len(rows)):
        orig = score_text(model, vocab, t, max_len)
        if y==1: # only attack malicious samples
            adv_text = make_inserted(t, api_pool, insert_n=args.insert_n)
            adv_score = score_text(model, vocab, adv_text, max_len)
            success = adv_score < 0.5  # misclassified as benign
            results.append({"id": i, "orig_score": orig, "adv_score": adv_score, "success": success})
        else:
            # skipping benign
            results.append({"id": i, "orig_score": orig, "adv_score": None, "success": None})
    dt = time.time()-t0
    out = {"meta": {"n": len(rows), "time_s": dt, "insert_n": args.insert_n}, "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w") as f:
        json.dump(out, f, indent=2)
    print("Wrote results to", args.out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--api_candidates", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--insert_n", type=int, default=100)
    p.add_argument("--sample_limit", type=int, default=None)
    p.add_argument("--max_workers", type=int, default=1)
    args = p.parse_args()
    run(args)

if __name__ == "__main__":
    main()
