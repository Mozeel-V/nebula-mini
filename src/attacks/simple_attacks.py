#!/usr/bin/env python3
"""
Simple problem-space attacks (insertion / replacement).
- Samples APIs from a candidate pool (built by api_candidates.py)
- Supports --sample_limit and --insert_n
- Optional --api_candidates <json> for diversity
- Optional --saliency_positions to integrate saliency suggestions (stub hook)
"""
import json, argparse, random
from pathlib import Path

from tokenization.tokenizer import tokenize, tokens_to_ids

def sample_event_from_api(api_token: str, idx: int = 0) -> str:
    """
    Maps an 'api:Name' token into a plausible event snippet used by our text format.
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

def insert_benign_events_text(text: str, api_pool, n: int = 50) -> str:
    pads = [sample_event_from_api(random.choice(api_pool), i) for i in range(n)]
    return text + " " + " ".join(pads)

def replace_api_tokens(text: str, api_pool, n_replace: int = 10) -> str:
    toks = text.split()
    api_pos = [i for i, t in enumerate(toks) if t.startswith("api:")]
    if not api_pos:
        return text
    for _ in range(min(n_replace, len(api_pos))):
        pos = random.choice(api_pos)
        toks[pos] = sample_event_from_api(random.choice(api_pool))
    return " ".join(toks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_file", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument("--api_candidates", default=None, help="JSON file produced by api_candidates.py")
    p.add_argument("--strategy", choices=["insert","replace"], default="insert")
    p.add_argument("--insert_n", type=int, default=100)
    p.add_argument("--sample_limit", type=int, default=None, help="limit number of samples to attack (shuffle applied)")
    p.add_argument("--saliency_positions", default=None, help="JSON mapping of sample id -> important token indices")     # saliency hook (optional): provide a JSON of {id: [positions]} to bias where we mutate
    args = p.parse_args()

    api_pool = None
    if args.api_candidates and Path(args.api_candidates).exists():
        api_pool = json.load(open(args.api_candidates, "r", encoding="utf-8"))
    if not api_pool:
        # minimal fallback if candidates missing
        api_pool = ["api:ReadFile", "api:CreateFileW", "api:WriteFile", "api:RegOpenKeyExW", "api:connect", "api:send", "api:recv", "api:Sleep"]

    # optional saliency positions (not strictly required for this script to run)
    sal_pos = {}
    if args.saliency_positions and Path(args.saliency_positions).exists():
        sal_pos = json.load(open(args.saliency_positions, "r", encoding="utf-8"))

    rows = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3:
                continue
            i, y, t = parts
            rows.append((int(i), int(y), t))

    random.shuffle(rows)
    if args.sample_limit is not None:
        rows = rows[: args.sample_limit]

    out_lines = []
    attacked = 0
    for i, y, t in rows:
        if y != 1:
            continue
        if args.strategy == "insert":
            adv = insert_benign_events_text(t, api_pool, n=args.insert_n)
        else:
            adv = replace_api_tokens(t, api_pool, n_replace=max(5, args.insert_n // 10))
        out_lines.append(f"{i}\t{y}\t{adv}")
        attacked += 1

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

    print(f"Wrote {len(out_lines)} adversarial samples to {out_path} (attacked {attacked} malicious traces).")

if __name__ == "__main__":
    main()