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

# ---------- Model / scoring ----------

def load_model(ckpt_path, vocab_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck["config"]
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
    state = ck.get("model", ck.get("model_state"))
    model.load_state_dict(state)
    model.eval()
    return model, vocab, cfg

@torch.no_grad()
def score_text(model, vocab, text, max_len):
    ids = tokens_to_ids(tokenize(text), vocab, max_len=max_len)
    x = torch.tensor([ids], dtype=torch.long)
    logits = model(x)
    prob_mal = torch.softmax(logits, dim=-1)[0, 1].item()
    return float(prob_mal)


# ---------- Attack utilities ----------

def sample_event_from_api(api_token: str, idx: int = 0) -> str:
    """
    Turn an 'api:Name' token into a plausible event snippet in our text format.
    """
    api = api_token.split("api:")[-1]
    low = api.lower()
    if low in ("readfile", "createfilew", "writefile"):
        return f"api:{api} path:C:\\\\Windows\\\\Temp\\\\pad{idx}.tmp"
    if low in ("connect", "send", "recv"):
        return f"api:{api} ip:127.0.0.1"
    if low.startswith("reg"):
        return f"api:{api} path:HKEY_LOCAL_MACHINE\\\\Software\\\\Vendor"
    if low.startswith("virtualalloc") or low.startswith("writeprocessmemory"):
        return f"api:{api} pid:1000"
    return f"api:{api}"

def propose_replacement(text: str, api_pool, n_replace=6, bias_positions=None):
    """
    Make a mutated candidate by replacing up to n_replace api:* tokens.
    If bias_positions provided, try to pick positions from that set first.
    """
    toks = text.split()
    api_pos = [i for i, t in enumerate(toks) if t.startswith("api:")]
    if not api_pos:
        return text  # no API tokens to mutate

    # Build candidate replacement positions
    positions = []
    if bias_positions:
        # keep only positions that actually exist as api tokens
        b = [i for i in bias_positions if i in api_pos]
        random.shuffle(b)
        positions.extend(b[:n_replace])
    # Fill remaining from general api positions
    if len(positions) < n_replace:
        rest = [i for i in api_pos if i not in positions]
        random.shuffle(rest)
        positions.extend(rest[: max(0, n_replace - len(positions))])

    # Apply replacements
    for j, pos in enumerate(positions):
        toks[pos] = sample_event_from_api(random.choice(api_pool), j)

    return " ".join(toks)


# ---------- Main search ----------

def run_ga(args):
    # Load model & vocab
    model, vocab, cfg = load_model(args.ckpt, args.vocab)

    # API pool
    if args.api_candidates and Path(args.api_candidates).exists():
        api_pool = json.load(open(args.api_candidates, "r", encoding="utf-8"))
    else:
        api_pool = ["api:ReadFile", "api:CreateFileW", "api:WriteFile", "api:RegOpenKeyExW",
                    "api:connect", "api:send", "api:recv", "api:Sleep"]

    # Optional saliency map: {id: [positions]}
    saliency = {}
    if args.saliency_json and Path(args.saliency_json).exists():
        saliency = json.load(open(args.saliency_json, "r", encoding="utf-8"))

    # Load dataset
    rows = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            i, y, t = parts
            rows.append((int(i), int(y), t))

    random.shuffle(rows)
    if args.sample_limit:
        rows = rows[: args.sample_limit]

    per_sample_results = []
    # To build an average convergence curve, we collect best score per iter for each sample
    per_iter_best_scores = [[] for _ in range(args.iters)]

    attacked = 0
    for sid, y, text in rows:
        if y != 1:
            continue  # attack only malicious

        attacked += 1
        orig_score = score_text(model, vocab, text, cfg["max_len"])

        # Start from the current text
        best_text = text
        best_score = orig_score

        # Per-sample convergence (best-so-far per iter)
        sample_best_curve = []

        bias_positions = None
        # saliency may store string or int keys; try both
        if args.saliency_json:
            bias_positions = saliency.get(str(sid), saliency.get(sid))

        for it in range(args.iters):
            candidates = []
            # Generate multiple candidate mutations this iteration
            for _ in range(args.candidates):
                cand = propose_replacement(
                    best_text, api_pool,
                    n_replace=args.n_replace,
                    bias_positions=bias_positions
                )
                cand_score = score_text(model, vocab, cand, cfg["max_len"])
                candidates.append((cand_score, cand))

            # pick the best candidate (lowest malware prob)
            cand_score, cand_text = min(candidates, key=lambda x: x[0])
            if cand_score < best_score:
                best_score, best_text = cand_score, cand_text

            # record sample best for this iteration
            sample_best_curve.append(best_score)

        # update global convergence accumulator
        for i, val in enumerate(sample_best_curve):
            per_iter_best_scores[i].append(val)

        per_sample_results.append({
            "id": sid,
            "label": 1,
            "orig_score": orig_score,
            "adv_score": best_score,
            "success": best_score < 0.5
        })

    # Average convergence across attacked samples
    convergence = []
    for i in range(args.iters):
        vals = per_iter_best_scores[i]
        if len(vals) == 0:
            convergence.append(None)
        else:
            convergence.append(sum(vals) / len(vals))

    out = {
        "attack": "ga",
        "iters": args.iters,
        "candidates": args.candidates,
        "n_replace": args.n_replace,
        "sample_limit": args.sample_limit,
        "results": per_sample_results,
        "convergence": convergence,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"GA attack done. Attacked {attacked} malicious traces.")
    print(f"Saved results to {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--api_candidates", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=25, help="GA/Hill-climb iterations")
    p.add_argument("--candidates", type=int, default=8, help="candidate mutations per iteration")
    p.add_argument("--n_replace", type=int, default=6, help="tokens replaced per candidate")
    p.add_argument("--sample_limit", type=int, default=None)
    p.add_argument("--saliency_json", default=None, help="Optional: bias positions per sample id")
    args = p.parse_args()
    run_ga(args)


if __name__ == "__main__":
    main()