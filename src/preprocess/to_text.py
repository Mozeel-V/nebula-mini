import os, json, argparse, random, sys
# Make src/ importable when running as a script
CURR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(CURR)
if SRC not in sys.path: sys.path.append(SRC)

from preprocess.field_filters import KEEP_FIELDS
from preprocess.normalizers import normalize
from tokenization.tokenizer import build_vocab

def flatten_event(ev):
    parts = []
    for k, v in ev.items():
        if k not in KEEP_FIELDS:
            continue
        if isinstance(v, dict):
            for kk, vv in v.items():
                parts.append(f"{k}:{kk}={normalize(str(vv))}")
        else:
            parts.append(f"{k}:{normalize(str(v))}")
    return " ".join(parts)

def load_and_flatten(raw_dir):
    files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.json')])
    texts, labels = [], []
    for fname in files:
        path = os.path.join(raw_dir, fname)
        with open(path, "r") as f:
            data = json.load(f)
        evs = data.get("events", [])
        seq = " ".join(flatten_event(e) for e in evs)
        texts.append(seq)
        label = 1 if "malware_" in fname else 0
        labels.append(label)
    return files, texts, labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--vocab_size", type=int, default=30000)
    ap.add_argument("--min_freq", type=int, default=1)
    args = ap.parse_args()

    files, texts, labels = load_and_flatten(args.raw_dir)

    vocab = build_vocab(texts, vocab_size=args.vocab_size, min_freq=args.min_freq)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    out_lines = []
    for i, (f, t, y) in enumerate(zip(files, texts, labels)):
        out_lines.append(f"{i}\t{y}\t{t}")
    with open(args.out_file, "w") as out:
        out.write("\n".join(out_lines))

    idxs = list(range(len(files)))
    random.seed(42); random.shuffle(idxs)
    n = len(idxs)
    n_train = int(0.7*n)
    n_val = int(0.15*n)
    splits = {"train": idxs[:n_train], "val": idxs[n_train:n_train+n_val], "test": idxs[n_train+n_val:]}
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, "w") as mf:
        json.dump({"splits": splits}, mf, indent=2)

    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/vocab.json", "w") as vf:
        json.dump(vocab, vf)

if __name__ == "__main__":
    main()
