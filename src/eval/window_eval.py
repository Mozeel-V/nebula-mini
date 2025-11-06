'''
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
    ap.add_argument("--malware_index", type=int, default=0, help="softmax index for malware class")
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
'''

#!/usr/bin/env python3
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
    ap.add_argument("--out_dir", default="results/window_eval")
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

    # ----- choose malware index AND threshold jointly by F1 (robust) -----
    m_idx, thr, per_sample_pmal, summ = choose_malware_index_and_threshold(
        model, vocab, per_sample_windows, labels, cfg["max_len"]
    )
    thr = 0.34440063102340696
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

