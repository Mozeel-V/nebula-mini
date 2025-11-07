#!/usr/bin/env python3
"""
Estimate per-API effectiveness:
For a sample of malware traces, for each API in api_candidates,
replace a high-saliency token with a generated event (api -> event string),
compute delta = orig_score - new_score (probability diff). Average deltas per API.

Output: checkpoints/api_effectiveness.json -> [{"api":"api:ReadFile","mean_delta":0.12,"count":50}, ...]
"""
import argparse, json, random, os, math
from pathlib import Path
import numpy as np
import tqdm

# ---- imports to use the model/tokenizer from the repo ----
import torch
from tokenization.tokenizer import tokenize, tokens_to_ids
from models.nebula_model import NebulaTiny

# ---- Model scorer helper (loads model/vocab once) ----
class ModelScorer:
    def __init__(self, ckpt_path: str, vocab_path: str, device=None):
        self.ckpt_path = ckpt_path
        self.vocab_path = vocab_path
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._load()

    def _load(self):
        ck = torch.load(self.ckpt_path, map_location="cpu")
        # handle different checkpoint layouts
        cfg = ck.get("config") or ck.get("cfg") or {}
        # fallback to values if not saved (try reasonable defaults)
        vocab = json.load(open(self.vocab_path, "r", encoding="utf-8"))
        vocab_size = cfg.get("vocab_size", len(vocab))
        d_model = cfg.get("d_model", 128)
        heads = cfg.get("heads", cfg.get("nhead", 4))
        layers = cfg.get("layers", cfg.get("num_layers", 1))
        ff = cfg.get("ff", cfg.get("dim_feedforward", 256))
        max_len = cfg.get("max_len", 512)

        # build model with same API as other scripts
        model = NebulaTiny(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=heads,
            num_layers=layers,
            dim_feedforward=ff,
            max_len=max_len,
            num_classes=2,
            chunk_size=cfg.get("chunk_size", 0)
        )
        # load state (supports keys "model" / "model_state" / "model_state_dict")
        state = ck.get("model", ck.get("model_state", ck.get("model_state_dict", None)))
        if state is None:
            # sometimes train_supervised saves with 'model_state' under top-level; check other keys
            # if nothing matches, try to use ck itself if it looks like state dict
            if isinstance(ck, dict) and any(k in ck for k in ("embed.weight", "cls.0.weight")):
                state = ck
        if state is None:
            raise RuntimeError(f"Cannot find model state in checkpoint {self.ckpt_path}. Available keys: {list(ck.keys())}")

        model.load_state_dict(state)
        model.to(self.device)
        model.eval()

        self.model = model
        self.vocab = vocab
        self.cfg = cfg

    def score_trace(self, trace_text: str, max_len: int = None):
        """
        Score a single flattened trace string and return P(malware) in [0,1].
        """
        max_len = max_len or int(self.cfg.get("max_len", 256))
        ids = tokens_to_ids(tokenize(trace_text), self.vocab, max_len=max_len)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(x)  # [1,2]
            probs = torch.softmax(logits, dim=-1)[0,1].item()  # P(malware)
        return float(probs)


# ---- utility functions ----
def load_api_pool(path):
    return json.load(open(path, "r", encoding="utf-8"))

def sample_event_from_api(api_token, idx=0):
    api = api_token.split("api:")[-1]
    low = api.lower()
    if low in ("readfile","createfilew","writefile"):
        return f"api:{api} path:C:\\\\Windows\\\\Temp\\\\pad{idx}.tmp"
    if low in ("connect","send","recv"):
        return f"api:{api} ip:127.0.0.1"
    if low.startswith("reg"):
        return f"api:{api} path:HKEY_LOCAL_MACHINE\\\\Software\\\\Vendor"
    return f"api:{api}"

def replace_token_at_index(flat_trace, token_idx, new_event):
    if " ||| " in flat_trace:
        parts = flat_trace.split(" ||| ")
        if 0 <= token_idx < len(parts):
            parts[token_idx] = new_event
        return " ||| ".join(parts)
    else:
        parts = flat_trace.split()
        if 0 <= token_idx < len(parts):
            parts[token_idx] = new_event
        return " ".join(parts)

def estimate_effects(args):
    api_pool = load_api_pool(args.api_pool)
    samples = []
    # read dataset TSV: id \t label \t trace
    with open(args.dataset, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            sid, lab, trace = ln.split("\t", 2)
            if int(lab) == 1:
                samples.append((sid, trace))
    if len(samples) == 0:
        print("No malware samples found in dataset.")
        return

    saliency = json.load(open(args.saliency_json, "r", encoding="utf-8"))
    if args.sample_limit and args.sample_limit < len(samples):
        samples = random.sample(samples, args.sample_limit)

    # create scorer
    scorer = ModelScorer(args.ckpt, args.vocab)

    api_stats = {a: [] for a in api_pool}

    for sid, trace in tqdm.tqdm(samples, desc="samples"):
        try:
            orig = scorer.score_trace(trace)
        except Exception as e:
            print("Scorer error:", e)
            return

        top_idxs = saliency.get(str(sid), saliency.get(int(sid), []))
        if not top_idxs:
            # fallback indices
            toks = trace.split(" ||| ") if " ||| " in trace else trace.split()
            top_idxs = list(range(min(len(toks), args.k)))
        top_idxs = top_idxs[:args.k]

        for api in api_pool:
            # sample a small number of positions per API to contain runtime
            positions = random.sample(top_idxs, min(len(top_idxs), args.per_api_positions))
            for pos in positions:
                new_event = sample_event_from_api(api, idx=random.randint(0,9999))
                adv_trace = replace_token_at_index(trace, pos, new_event)
                new_score = scorer.score_trace(adv_trace)
                # delta: positive means orig > new (we reduced mal prob)
                delta = float(orig) - float(new_score)
                api_stats[api].append(delta)

    out = []
    for api, deltas in api_stats.items():
        mean = float(np.mean(deltas)) if len(deltas) > 0 else 0.0
        out.append({"api": api, "mean_delta": mean, "count": len(deltas)})
    out_sorted = sorted(out, key=lambda x: x["mean_delta"], reverse=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_sorted, f, indent=2)
    print(f"Wrote api effectiveness ({len(out_sorted)}) -> {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="balanced TSV dataset (id \\t label \\t trace)")
    p.add_argument("--api_pool", required=True, help="checkpoints/api_candidates.json")
    p.add_argument("--saliency_json", required=True, help="saliency positions JSON (map sid->list of positions)")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--out", default="real-data/checkpoints/api_effectiveness.json")
    p.add_argument("--sample_limit", type=int, default=200, help="max malware samples to sample for estimation")
    p.add_argument("--k", type=int, default=32, help="top-k saliency positions to try per sample")
    p.add_argument("--per_api_positions", type=int, default=3, help="positions to try per api per sample")
    args = p.parse_args()
    estimate_effects(args)

if __name__ == "__main__":
    main()
