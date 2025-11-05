#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Window-level evaluation + per-sample class timelines.

What it does
------------
1) Splits each sample into sequential windows (event- or token-based).
2) Scores every window with the loaded checkpoint.
3) Picks a probability threshold on P(malware) (by F1 on the validation-like split).
4) Computes sample-level metrics with the rule: sample is malware if ANY window >= threshold.
5) Saves:
   - results/window_eval/window_eval.json (metrics + per-sample scores)
   - figures/per_sample_class/sample_<id>.png (0/1 class timeline per sample)
   - quick examples: goodware_time_series_class.png, malware_before_time_series_class.png
"""

import argparse, json, sys, math, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------- local imports ----------------
try:
    from tokenization.tokenizer import tokenize, tokens_to_ids
    from models.nebula_model import NebulaTiny
except Exception:
    CURR = Path(__file__).resolve().parents[2] / "src"
    if str(CURR) not in sys.path:
        sys.path.append(str(CURR))
    from tokenization.tokenizer import tokenize, tokens_to_ids  # type: ignore
    from models.nebula_model import NebulaTiny  # type: ignore

# ---------------- windowing ----------------
def windows_by_events(text, events_per_window=16, stride_events=4):
    toks = text.split()
    api_idx = [i for i, t in enumerate(toks) if t.startswith("api:")]
    if not api_idx:
        # fallback: token windows
        return windows_by_tokens(text, tokens_per_window=events_per_window*8,
                                 stride_tokens=max(1, stride_events)*8)
    starts = api_idx + [len(toks)]
    events = [" ".join(toks[starts[i]:starts[i+1]]) for i in range(len(starts)-1)]
    wins = []
    for i in range(0, len(events), stride_events):
        seg = events[i:i+events_per_window]
        if not seg: break
        wins.append(" ".join(seg))
    return wins

def windows_by_tokens(text, tokens_per_window=256, stride_tokens=64):
    toks = text.split()
    wins = []
    for i in range(0, len(toks), stride_tokens):
        seg = toks[i:i+tokens_per_window]
        if not seg: break
        wins.append(" ".join(seg))
    return wins

# ---------------- model I/O ----------------
def load_model_and_vocab(ckpt_path, vocab_path):
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

def logits_to_probs(logits_2):
    # logits_2: [W,2]
    logits_2 = logits_2 - logits_2.max(axis=1, keepdims=True)  # stability
    e = np.exp(logits_2)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

def score_windows(model, vocab, windows, max_len):
    """Return probs [W,2] for a list of window texts."""
    out = []
    for w in windows:
        ids = tokens_to_ids(tokenize(w), vocab, max_len=max_len)
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            lo = model(x)[0].cpu().numpy()  # [2]
        out.append(lo)
    if not out:
        return np.zeros((0, 2), dtype=np.float32)
    logits = np.vstack(out)
    return logits_to_probs(logits)

# ---------------- threshold + metrics ----------------
def pick_threshold_f1(per_sample_pmal, labels):
    """
    per_sample_pmal: dict[sid] -> list of P(mal) per window
    labels: dict[sid] -> 0/1
    Returns: best_thr, best_f1, tp, fp, fn
    """
    # gather candidates
    vals = []
    for sid, arr in per_sample_pmal.items():
        vals.extend(arr)
    if not vals:
        return 0.5, 0.0, 0, 0, 0
    candidates = np.unique(np.asarray(vals))
    if candidates.size > 400:
        # subsample thresholds uniformly
        candidates = np.linspace(candidates.min(), candidates.max(), 400)

    best_f1, best_thr, best_tuple = -1.0, 0.5, (0, 0, 0)
    for t in candidates:
        tp=fp=tn=fn=0
        for sid, arr in per_sample_pmal.items():
            y = labels[sid]
            pred = 1 if (len(arr)>0 and np.max(arr) >= t) else 0
            if pred==1 and y==1: tp += 1
            elif pred==1 and y==0: fp += 1
            elif pred==0 and y==0: tn += 1
            else: fn += 1
        prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
        f1 = 2*prec*rec/(prec+rec+1e-12)
        if f1 > best_f1:
            best_f1, best_thr, best_tuple = f1, float(t), (tp, fp, fn)
    return best_thr, best_f1, *best_tuple

# ---------------- malware index auto-detect ----------------
def detect_malware_index(model, vocab, per_sample_windows, labels, max_len, sample_cap=60):
    def collect_means(col):
        mal, ben = [], []
        n=0
        for sid, wins in per_sample_windows.items():
            if n>=sample_cap: break
            if not wins: continue
            p = score_windows(model, vocab, wins, max_len)  # [W,2]
            if labels[sid]==1:
                mal.extend(p[:, col])
            else:
                ben.extend(p[:, col])
            n += 1
        if len(mal)==0 or len(ben)==0: return -1e9
        return float(np.mean(mal) - np.mean(ben))
    d0 = collect_means(0)
    d1 = collect_means(1)
    return 0 if d0 >= d1 else 1

def choose_malware_index_and_threshold(model, vocab, per_sample_windows, labels, max_len):
    """
    Try both columns (0/1) as 'malware'. For each, compute P(mal) per sample,
    choose the threshold that maximizes sample-level F1 (any-window rule).
    Return (best_m_idx, best_thr, best_summary_dict).
    """
    import numpy as np

    def probs_for_idx(m_idx):
        per_sample_pmal = {}
        for sid, wins in per_sample_windows.items():
            # get probs [W,2]
            out = []
            for w in wins:
                ids = tokens_to_ids(tokenize(w), vocab, max_len=max_len)
                x = torch.tensor([ids], dtype=torch.long)
                with torch.no_grad():
                    lo = model(x)[0].cpu().numpy()
                out.append(lo)
            if out:
                L = np.vstack(out)
                L = L - L.max(axis=1, keepdims=True)
                e = np.exp(L); p = e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)
                per_sample_pmal[sid] = p[:, m_idx].tolist()
            else:
                per_sample_pmal[sid] = []
        return per_sample_pmal

    def pick_thr(per_sample_pmal):
        vals = [v for arr in per_sample_pmal.values() for v in arr]
        if not vals: return 0.5, (0,0,0,0), 0.0
        cand = np.unique(np.asarray(vals))
        if cand.size > 400:
            cand = np.linspace(cand.min(), cand.max(), 400)
        best = (-1.0, 0.5, (0,0,0,0))  # f1, thr, (tp,fp,tn,fn)
        for t in cand:
            tp=fp=tn=fn=0
            for sid, arr in per_sample_pmal.items():
                y = labels[sid]
                pred = 1 if (len(arr)>0 and np.max(arr) >= t) else 0
                if pred==1 and y==1: tp+=1
                elif pred==1 and y==0: fp+=1
                elif pred==0 and y==0: tn+=1
                else: fn+=1
            prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
            f1 = 2*prec*rec/(prec+rec+1e-12)
            if f1 > best[0]:
                best = (f1, float(t), (tp,fp,tn,fn))
        return best[1], best[2], best[0]

    results = []
    for m_idx in (0, 1):
        pmal = probs_for_idx(m_idx)
        thr, (tp,fp,tn,fn), f1 = pick_thr(pmal)
        results.append((f1, m_idx, thr, pmal, tp,fp,tn,fn))

    # pick by best F1
    results.sort(reverse=True, key=lambda x: x[0])
    f1, m_idx, thr, per_sample_pmal, tp,fp,tn,fn = results[0]
    acc = (tp+tn)/max(1,len(labels))
    prec = tp/max(1,tp+fp); rec = tp/max(1,tp+fn)
    summary = {"threshold": thr, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
               "accuracy": acc, "precision": prec, "recall": rec}
    return m_idx, thr, per_sample_pmal, summary


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out_dir", default="results/window_eval_plot")
    ap.add_argument("--window_unit", choices=["event", "token"], default="event")
    ap.add_argument("--events_per_window", type=int, default=16)
    ap.add_argument("--stride_events", type=int, default=4)
    ap.add_argument("--tokens_per_window", type=int, default=256)
    ap.add_argument("--stride_tokens", type=int, default=64)
    ap.add_argument("--sample_limit", type=int, default=0, help="0 = all")
    ap.add_argument("--malware_index", default="0", help="0, 1, or 'auto'")
    args = ap.parse_args()

    model, vocab, cfg = load_model_and_vocab(args.ckpt, args.vocab)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "window_figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    # ----- read data -----
    rows = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t", 2)
            if len(p) != 3: continue
            sid, y, txt = int(p[0]), int(p[1]), p[2]
            rows.append((sid, y, txt))
    if args.sample_limit and args.sample_limit > 0:
        rows = rows[:args.sample_limit]

    # ----- build windows per sample (temporal order) -----
    per_sample_windows = {}
    labels = {}
    for sid, y, txt in rows:
        if args.window_unit == "event":
            wins = windows_by_events(txt, args.events_per_window, args.stride_events)
        else:
            wins = windows_by_tokens(txt, args.tokens_per_window, args.stride_tokens)
        per_sample_windows[sid] = wins
        labels[sid] = y

    '''
    # ----- decide malware index -----
    if str(args.malware_index).lower() == "auto":
        m_idx = detect_malware_index(model, vocab, per_sample_windows, labels, cfg["max_len"])
        print(f"[INFO] Auto-detected malware class index = {m_idx}")
    else:
        m_idx = int(args.malware_index)
        print(f"[INFO] Using malware class index = {m_idx}")

    # ----- score per-window, collect P(mal) -----
    per_sample_pmal = {}
    for sid, wins in per_sample_windows.items():
        probs = score_windows(model, vocab, wins, cfg["max_len"])  # [W,2]
        p_mal = probs[:, m_idx] if probs.size else np.zeros((0,), dtype=np.float32)
        per_sample_pmal[sid] = p_mal.tolist()

    # ----- choose threshold (best sample-level F1) -----
    thr, best_f1, tp, fp, fn = pick_threshold_f1(per_sample_pmal, labels)
    tn = len(labels) - tp - fp - fn
    acc = (tp + tn) / max(1, len(labels))
    prec = tp / max(1, tp+fp)
    rec = tp / max(1, tp+fn)

    print(f"[INFO] Auto-picked threshold = {thr:.3f}  (best F1={best_f1:.3f}, tp={tp}, fp={fp}, fn={fn})")
    print(f"[RESULT] Sample-level summary: accuracy={acc:.3f}, precision={prec:.3f}, recall={rec:.3f}, f1={best_f1:.3f}, tp={tp}, fp={fp}, tn={tn}, fn={fn}")

    # ----- save JSON summary -----
    summary = {
        "threshold": thr,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": best_f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "malware_index": m_idx,
        "params": {
            "window_unit": args.window_unit,
            "events_per_window": args.events_per_window,
            "stride_events": args.stride_events,
            "tokens_per_window": args.tokens_per_window,
            "stride_tokens": args.stride_tokens,
            "sample_limit": args.sample_limit,
        },
        "samples": {
            str(sid): {
                "label": int(labels[sid]),
                "scores": per_sample_pmal[sid],
                "n_windows": len(per_sample_pmal[sid]),
            } for sid in per_sample_pmal
        }
    }
    json_path = out_dir / "window_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Wrote {json_path}")
    '''

    # ----- choose malware index AND threshold jointly by F1 (robust) -----
    m_idx, thr, per_sample_pmal, summ = choose_malware_index_and_threshold(
        model, vocab, per_sample_windows, labels, cfg["max_len"]
    )
    print(
        f"[INFO] Selected malware_index={m_idx} | thr={thr:.3f} | "
        f"F1={summ['f1']:.3f} (tp={summ['tp']}, fp={summ['fp']}, tn={summ['tn']}, fn={summ['fn']})"
    )

    # unpack metrics for convenience
    best_f1 = summ["f1"]
    tp, fp, tn, fn = summ["tp"], summ["fp"], summ["tn"], summ["fn"]
    acc, prec, rec = summ["accuracy"], summ["precision"], summ["recall"]

    # ----- save JSON summary -----
    summary = {
        "threshold": thr,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": best_f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "malware_index": m_idx,
        "params": {
            "window_unit": args.window_unit,
            "events_per_window": args.events_per_window,
            "stride_events": args.stride_events,
            "tokens_per_window": args.tokens_per_window,
            "stride_tokens": args.stride_tokens,
            "sample_limit": args.sample_limit,
        },
        "samples": {
            str(sid): {
                "label": int(labels[sid]),
                "scores": per_sample_pmal[sid],
                "n_windows": len(per_sample_pmal[sid]),
            } for sid in per_sample_pmal
        }
    }
    json_path = out_dir / "window_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Wrote {json_path}")


    # ---------------- PLOTTING: Per-sample CLASS timelines ----------------
    MALWARE_INDEX = m_idx  # for clarity below
    class_dir = fig_dir / "per_sample_class"
    class_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Generating class-timeline plots to {class_dir}")

    def win_logits_for_sample(wins):
        arr = []
        for w in wins:
            ids = tokens_to_ids(tokenize(w), vocab, max_len=cfg["max_len"])
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                lo = model(x)[0].cpu().numpy()  # [2]
            arr.append(lo)
        if not arr: return np.zeros((0,2), dtype=np.float32)
        return np.vstack(arr)

    def probs_from_logits(logits):
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)

    for sid in sorted(per_sample_windows.keys()):
        wins = per_sample_windows[sid]  # sequential order
        if not wins:
            continue
        logits = win_logits_for_sample(wins)  # [W,2]
        probs  = probs_from_logits(logits)
        p_mal  = probs[:, MALWARE_INDEX] if probs.size else np.zeros((0,))
        pred_cls = (p_mal >= thr).astype(np.int32)  # 1=malware, 0=benign

        xs = np.arange(len(pred_cls))
        plt.figure(figsize=(10, 2.8))
        plt.step(xs, pred_cls, where="mid", linewidth=1.2)
        plt.scatter(xs, pred_cls, s=12)
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        plt.text(0.01, 1.02, f"thr P(mal)={thr:.3f}", transform=plt.gca().transAxes, fontsize=8)
        plt.ylim(-0.2, 1.2)
        plt.yticks([0, 1], ["Benign", "Malware"])
        lbl = "Malware" if labels[sid] == 1 else "Benign"
        plt.title(f"Sample {sid} | Label: {lbl}")
        plt.xlabel("Window sequence index")
        plt.ylabel("Predicted class")
        plt.grid(axis="x", alpha=0.2)
        plt.tight_layout()
        outpng = class_dir / f"sample_{sid}.png"
        plt.savefig(outpng, bbox_inches="tight", dpi=130)
        plt.close()

    print(f"[INFO] Saved per-sample class timelines to {class_dir}")

    # ---------------- quick canonical examples ----------------
    def pick_examples(per_sample_pmal, labels):
        ben_cands = [(sid, max(p) if p else -1.0) for sid, p in per_sample_pmal.items() if labels[sid]==0 and p]
        mal_cands = [(sid, max(p) if p else -1.0) for sid, p in per_sample_pmal.items() if labels[sid]==1 and p]
        ben_sid = min(ben_cands, key=lambda x: x[1])[0] if ben_cands else None
        mal_sid = max(mal_cands, key=lambda x: x[1])[0] if mal_cands else None
        return ben_sid, mal_sid

    ben_sid, mal_sid = pick_examples(per_sample_pmal, labels)

    def quick_class_plot(sid, fname, title):
        wins = per_sample_windows[sid]
        logits = win_logits_for_sample(wins)
        probs  = probs_from_logits(logits)
        p_mal  = probs[:, MALWARE_INDEX] if probs.size else np.zeros((0,))
        pred_cls = (p_mal >= thr).astype(np.int32)
        xs = np.arange(len(pred_cls))
        plt.figure(figsize=(10, 2.8))
        plt.step(xs, pred_cls, where="mid", linewidth=1.2)
        plt.scatter(xs, pred_cls, s=12)
        plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        plt.ylim(-0.2, 1.2); plt.yticks([0,1], ["Benign","Malware"])
        plt.xlabel("Window sequence index"); plt.ylabel("Predicted class")
        plt.title(title); plt.tight_layout()
        plt.savefig(Path(fig_dir)/fname, bbox_inches="tight", dpi=130); plt.close()

    if ben_sid is not None:
        quick_class_plot(ben_sid, "goodware_time_series_class.png",
                         f"Goodware sample (sid={ben_sid}) — window classes")
    if mal_sid is not None:
        quick_class_plot(mal_sid, "malware_before_time_series_class.png",
                         f"Malware sample (sid={mal_sid}) — window classes")

    print(f"[INFO] Also wrote quick example class plots to {fig_dir}")

    # ---------------- optional: also dump probability/signed plots (off by default) ----------------
    # To enable, set these to True:
    MAKE_PROB_PLOTS = False
    MAKE_SIGNED_PLOTS = False  # signed = P(mal) - P(ben)

    if MAKE_PROB_PLOTS:
        prob_dir = fig_dir / "per_sample_prob"
        prob_dir.mkdir(parents=True, exist_ok=True)
        for sid in sorted(per_sample_windows.keys()):
            wins = per_sample_windows[sid]
            if not wins: continue
            logits = win_logits_for_sample(wins)
            probs  = probs_from_logits(logits)
            p_mal  = probs[:, MALWARE_INDEX] if probs.size else np.zeros((0,))
            xs = np.arange(len(p_mal))
            plt.figure(figsize=(10, 2.8))
            plt.plot(xs, p_mal, marker="o", linewidth=1.2)
            plt.axhline(thr, color="gray", linestyle="--", linewidth=0.8)
            plt.ylim(-0.05, 1.05)
            plt.xlabel("Window sequence index"); plt.ylabel("P(malware)")
            lbl = "Malware" if labels[sid] == 1 else "Benign"
            plt.title(f"Sample {sid} | Label: {lbl}")
            plt.tight_layout()
            plt.savefig(prob_dir / f"sample_{sid}.png", bbox_inches="tight", dpi=130)
            plt.close()
        print(f"[INFO] Saved per-sample probability plots to {prob_dir}")

    if MAKE_SIGNED_PLOTS:
        signed_dir = fig_dir / "per_sample_signed"
        signed_dir.mkdir(parents=True, exist_ok=True)
        for sid in sorted(per_sample_windows.keys()):
            wins = per_sample_windows[sid]
            if not wins: continue
            logits = win_logits_for_sample(wins)
            probs  = probs_from_logits(logits)
            p_mal  = probs[:, MALWARE_INDEX] if probs.size else np.zeros((0,))
            p_ben  = 1.0 - p_mal
            signed = p_mal - p_ben  # = 2*p_mal - 1
            xs = np.arange(len(signed))
            plt.figure(figsize=(10, 2.8))
            plt.plot(xs, signed, marker="o", linewidth=1.2)
            plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
            plt.axhline((thr*2.0)-1.0, color="black", linestyle=":", linewidth=0.8)
            plt.ylim(-1.05, 1.05)
            plt.xlabel("Window sequence index"); plt.ylabel("Signed confidence (+mal / -ben)")
            lbl = "Malware" if labels[sid] == 1 else "Benign"
            plt.title(f"Sample {sid} | Label: {lbl}")
            plt.tight_layout()
            plt.savefig(signed_dir / f"sample_{sid}.png", bbox_inches="tight", dpi=130)
            plt.close()
        print(f"[INFO] Saved per-sample signed-confidence plots to {signed_dir}")

if __name__ == "__main__":
    main()
