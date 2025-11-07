#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
from saliency_weighted_sampler import Sampler

def load_dataset_tsv(path):
    out=[]
    with open(path,"r",encoding="utf-8") as f:
        for ln in f:
            sid,lab,trace = ln.rstrip("\n").split("\t",2)
            out.append((sid,int(lab),trace))
    return out

def replace_token(flat, idx, ev):
    if " ||| " in flat:
        parts = flat.split(" ||| ")
        if 0 <= idx < len(parts):
            parts[idx] = ev
        return " ||| ".join(parts)
    else:
        parts = flat.split()
        if 0 <= idx < len(parts):
            parts[idx] = ev
        return " ".join(parts)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--saliency_json", required=True)
    p.add_argument("--api_effect", required=True)
    p.add_argument("--api_pool", required=True)
    p.add_argument("--out", default="results/attacks/saliency_sampler_adversarial.tsv")
    p.add_argument("--n_replace", type=int, default=6)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--sample_limit", type=int, default=100)
    args=p.parse_args()

    sampler = Sampler(args.api_pool, api_effectiveness=args.api_effect, benign_counts=None, temp=0.9)
    ds = load_dataset_tsv(args.dataset)
    salmap = json.load(open(args.saliency_json,"r",encoding="utf-8"))

    adv_lines=[]
    cnt=0
    for sid,lab,trace in ds:
        if lab!=1:
            adv_lines.append((sid,lab,trace))
            continue
        # sampleLimit
        if args.sample_limit and cnt >= args.sample_limit:
            adv_lines.append((sid,lab,trace))
            continue
        cnt+=1
        top = salmap.get(str(sid), salmap.get(int(sid), []))[:args.k]
        if not top:
            # fallback: first tokens
            toks = trace.split(" ||| ") if " ||| " in trace else trace.split()
            top = list(range(min(len(toks), args.k)))
        chosen = random.sample(top, min(len(top), args.n_replace))
        adv = trace
        for i,pos in enumerate(chosen):
            ev = sampler.sample_api_event(idx=i)
            adv = replace_token(adv, pos, ev)
        adv_lines.append((sid,lab,adv))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f:
        for sid,lab,trace in adv_lines:
            f.write(f"{sid}\t{lab}\t{trace}\n")
    print("Wrote", args.out)

if __name__=="__main__":
    main()
