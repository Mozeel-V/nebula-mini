#!/usr/bin/env python3
"""
Window-wise, sample-level evaluation + propagation analysis (robust version).

Key features:
- Event- or token-based windowing with overlap
- Per-window scoring -> per-sample decision
- Auto threshold sweep (maximize sample-level F1)
- Debug stats + figures
- Propagation-style risky benign windows

Outputs:
- results/window_eval/window_eval.json
- figures/goodware_time_series.png
- figures/malware_before_time_series.png
- figures/propagation_topN.png
"""

import argparse, json, hashlib, sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import torch

# ---------- project imports ----------
try:
    from tokenization.tokenizer import tokenize, tokens_to_ids
    from models.nebula_model import NebulaTiny
except Exception:
    CUR = Path(__file__).resolve().parents[2] / "src"
    if str(CUR) not in sys.path:
        sys.path.insert(0, str(CUR))
    from tokenization.tokenizer import tokenize, tokens_to_ids  # type: ignore
    from models.nebula_model import NebulaTiny  # type: ignore


# ---------- helpers ----------
def load_model(ckpt, vocab_path):
    ck = torch.load(ckpt, map_location="cpu")
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
def score_text(model, vocab, text, max_len, malware_index=1):
    ids = tokens_to_ids(tokenize(text), vocab, max_len=max_len)
    x = torch.tensor([ids], dtype=torch.long)
    logits = model(x)
    prob = torch.softmax(logits, dim=-1)[0, malware_index].item()
    return float(prob)

def windows_by_events(text, events_per_window=32, stride_events=None):
    """Split flattened trace into windows of N events; event begins at token starting with 'api:'."""
    toks = text.split()
    api_idx = [i for i, t in enumerate(toks) if t.startswith("api:")]
    if not api_idx:
        # fallback to token-based if no API markers
        return windows_by_tokens(text, tokens_per_window=events_per_window*8, stride_tokens=(stride_events or events_per_window)*8)

    # create list of event strings
    event_starts = api_idx + [len(toks)]
    events = [" ".join(toks[event_starts[i]:event_starts[i+1]]) for i in range(len(event_starts)-1)]

    stride_events = stride_events or events_per_window
    wins = []
    for i in range(0, len(events), stride_events):
        seg = events[i:i+events_per_window]
        if seg:
            wins.append(" ".join(seg))
    return wins

def windows_by_tokens(text, tokens_per_window=256, stride_tokens=None):
    toks = text.split()
    stride_tokens = stride_tokens or tokens_per_window
    wins = []
    for i in range(0, len(toks), stride_tokens):
        seg = toks[i:i+tokens_per_window]
        if seg:
            wins.append(" ".join(seg))
    return wins

def auto_pick_threshold(per_sample_scores, labels):
    """
    Choose threshold that maximizes sample-level F1 using
    candidates drawn from score distribution.
    per_sample_scores: dict[sid] -> list of window scores
    labels: dict[sid] -> 0/1
    """
    # candidate thresholds: unique max-window scores + a dense grid
    maxima = [max(v) if v else 0.0 for v in per_sample_scores.values()]
    grid = np.unique(np.clip(np.array(maxima + [i/100 for i in range(5, 96)], dtype=float), 0, 1))
    best = (0.0, 0.5, 0, 0, 0)  # (f1, thr, tp, fp, fn)

    for thr in grid:
        tp = fp = fn = 0
        for sid, scores in per_sample_scores.items():
            pred = 1 if any(s > thr for s in scores) else 0
            y = labels[sid]
            if pred == 1 and y == 1: tp += 1
            elif pred == 1 and y == 0: fp += 1
            elif pred == 0 and y == 1: fn += 1
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2 * prec * rec / (prec + rec + 1e-12)
        if f1 > best[0]:
            best = (f1, float(thr), tp, fp, fn)

    f1, thr, tp, fp, fn = best

    # Fallback: if best F1 is still ~0 (malware never rises), set thr to
    # 80th percentile of benign maxima to force some separation.
    if f1 < 1e-6:
        max_mal = [max(v) for sid, v in per_sample_scores.items() if labels[sid] == 1 and v]
        max_ben = [max(v) for sid, v in per_sample_scores.items() if labels[sid] == 0 and v]
        if max_ben:
            p80 = float(np.percentile(max_ben, 80))
            thr = max(0.05, min(0.95, p80))
        else:
            thr = 0.3
    return thr, best

def debug_stats(per_sample, labels):
    mal_max = [max(v) for sid, v in per_sample.items() if labels[sid]==1 and v]
    ben_max = [max(v) for sid, v in per_sample.items() if labels[sid]==0 and v]
    def stats(arr):
        if not arr: return (0,0,0,0)
        return (float(np.min(arr)), float(np.median(arr)), float(np.max(arr)), float(np.mean(arr)))
    m = stats(mal_max)
    b = stats(ben_max)
    print(f"[DEBUG] Malware max window scores min/med/max/mean = {m}")
    print(f"[DEBUG] Benign  max window scores min/med/max/mean = {b}")
    return m, b


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_file", required=True, help="processed dataset: id\\tlabel\\ttext")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out_dir", default="results/window_eval")
    ap.add_argument("--window_unit", choices=["event","token"], default="event")
    ap.add_argument("--events_per_window", type=int, default=32)
    ap.add_argument("--stride_events", type=int, default=None)
    ap.add_argument("--tokens_per_window", type=int, default=256)
    ap.add_argument("--stride_tokens", type=int, default=None)
    ap.add_argument("--sample_limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None, help="optional fixed threshold; if omitted, auto-picked")
    ap.add_argument("--malware_index", type=int, default=1, help="softmax index for malware class")
    args = ap.parse_args()

    model, vocab, cfg = load_model(args.ckpt, args.vocab)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir.parent / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- load data
    rows = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) != 3: continue
            i, y, t = parts
            rows.append((int(i), int(y), t))
    if args.sample_limit: rows = rows[:args.sample_limit]
    print(f"[INFO] Loaded {len(rows)} samples")

    # ---- windowing + scoring
    per_sample_scores = {}
    per_sample_windows = {}
    labels = {}
    win_hash_occ = Counter()
    win_hash_examples = defaultdict(list)

    for sid, y, text in rows:
        labels[sid] = y
        if args.window_unit == "event":
            wins = windows_by_events(text, events_per_window=args.events_per_window, stride_events=args.stride_events)
        else:
            wins = windows_by_tokens(text, tokens_per_window=args.tokens_per_window, stride_tokens=args.stride_tokens)

        # ensure we have at least one window (fallback by tokens if needed)
        if not wins:
            wins = windows_by_tokens(text, tokens_per_window=max(64, args.tokens_per_window))

        scores = [score_text(model, vocab, w, max_len=cfg["max_len"], malware_index=args.malware_index) for w in wins]
        per_sample_scores[sid] = scores
        per_sample_windows[sid] = wins

        # track duplicates for propagation analysis
        for w in wins:
            h = hashlib.sha1(w.encode("utf-8")).hexdigest()
            win_hash_occ[h] += 1
            if len(win_hash_examples[h]) < 3:
                win_hash_examples[h].append((sid, w))

    # ---- debug stats
    m_stats, b_stats = debug_stats(per_sample_scores, labels)

    # ---- choose threshold
    if args.threshold is None:
        thr, (best_f1, _, tp_s, fp_s, fn_s) = auto_pick_threshold(per_sample_scores, labels)
        print(f"[INFO] Auto-picked threshold = {thr:.3f}  (best F1={best_f1:.3f}, tp={tp_s}, fp={fp_s}, fn={fn_s})")
    else:
        thr = float(args.threshold)
        print(f"[INFO] Using fixed threshold = {thr:.3f}")

    # ---- sample-level decision + metrics
    tp = fp = tn = fn = 0
    sample_preds = {}
    for sid, scores in per_sample_scores.items():
        pred = 1 if any(s > thr for s in scores) else 0
        sample_preds[sid] = pred
        y = labels[sid]
        if pred==1 and y==1: tp += 1
        elif pred==1 and y==0: fp += 1
        elif pred==0 and y==0: tn += 1
        elif pred==0 and y==1: fn += 1
    prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    acc = (tp+tn)/max(1, tp+tn+fp+fn)
    summary = {"threshold": thr, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
               "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    print("[RESULT] Sample-level summary:", summary)

    # ---- risky benign windows (propagation perspective)
    risky = []
    for h, occ in win_hash_occ.items():
        sid_ex, w_ex = win_hash_examples[h][0]
        y = labels[sid_ex]
        s = score_text(model, vocab, w_ex, max_len=cfg["max_len"], malware_index=args.malware_index)
        if y == 0 and s > thr:
            risky.append((h, occ, s))
    risky.sort(key=lambda x: (x[1], x[2]), reverse=True)
    risky_top = [{"hash": h, "occ": int(occ), "score": float(s)} for (h, occ, s) in risky[:50]]

    # ---- save JSON
    out = {
        "summary": summary,
        "malware_max_stats": {"min": m_stats[0], "median": m_stats[1], "max": m_stats[2], "mean": m_stats[3]},
        "benign_max_stats":  {"min": b_stats[0], "median": b_stats[1], "max": b_stats[2], "mean": b_stats[3]},
        "samples": { int(sid): {"label": int(labels[sid]), "scores": [float(x) for x in per_sample_scores[sid]]}
                     for sid in per_sample_scores },
        "risky_windows_top": risky_top
    }
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(args.out_dir)/"window_eval.json", "w"), indent=2)
    print(f"[INFO] Wrote {Path(args.out_dir)/'window_eval.json'}")

    # ---- Generating per-sample time-series plots ----
    all_plots_dir = Path(fig_dir) / "per_sample_window"
    all_plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Generating time-series plots for {len(per_sample_scores)} samples...")

    for sid, scores in per_sample_scores.items():
        xs = list(range(len(scores)))
        y = scores
        label = labels[sid]
        plt.figure(figsize=(8, 2.5))
        plt.plot(xs, y, marker="o", linewidth=1.2, color="tab:red" if label == 1 else "tab:blue")
        plt.axhline(thr, color="gray", linestyle="--", linewidth=0.8)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Window index")
        plt.ylabel("Malware probability")
        plt.title(f"Sample {sid} | Label: {'Malware' if label==1 else 'Benign'}")
        plt.tight_layout()
        plt.savefig(all_plots_dir / f"sample_{sid}.png", dpi=120)
        plt.close()

    print(f"[INFO] Saved {len(per_sample_scores)} individual plots to {all_plots_dir}")

    # ---- plotting: pick examples
    def pick_benign(per_sample_scores, labels, bound=0.3):
        for sid, sc in per_sample_scores.items():
            if labels[sid]==0 and sc and max(sc) < bound:
                return sid
        return None

    def pick_malware(per_sample_scores, labels, bound=0.6):
        # prefer one that crosses threshold
        for sid, sc in per_sample_scores.items():
            if labels[sid]==1 and sc and max(sc) > bound:
                return sid
        # fallback: highest max
        best_sid, best_val = None, -1
        for sid, sc in per_sample_scores.items():
            if labels[sid]==1 and sc:
                v = max(sc)
                if v > best_val:
                    best_val, best_sid = v, sid
        return best_sid

    ben_sid = pick_benign(per_sample_scores, labels)
    mal_sid = pick_malware(per_sample_scores, labels)

    def plot_series(scores, title, fname, thr_line):
        xs = list(range(len(scores)))
        plt.figure(figsize=(10, 3))
        plt.plot(xs, scores, marker="o", linestyle="-", color="black")
        plt.axhline(thr_line, linestyle="--", color="gray")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Window index (time)")
        plt.ylabel("Predicted probability")
        plt.title(title)
        plt.grid(axis="y", alpha=0.3)
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(fig_dir) / fname, bbox_inches="tight")
        plt.close()

    if ben_sid is not None:
        plot_series(per_sample_scores[ben_sid],
                    f"Goodware sample (sid={ben_sid})", "goodware_time_series.png", thr)
        print(f"[INFO] Wrote {fig_dir}\\goodware_time_series.png (sid={ben_sid})")
    else:
        print("[WARN] No benign sample available for plotting")

    if mal_sid is not None:
        plot_series(per_sample_scores[mal_sid],
                    f"Malware sample (sid={mal_sid})", "malware_before_time_series.png", thr)
        print(f"[INFO] Wrote {fig_dir}\\malware_before_time_series.png (sid={mal_sid})")
    else:
        print("[WARN] No malware sample available for plotting")

    print(f"[INFO] Saved figures to {fig_dir}")

if __name__ == "__main__":
    main()
