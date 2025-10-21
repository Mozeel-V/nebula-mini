import argparse, json, torch, os, sys
CURR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(CURR)
if SRC not in sys.path: sys.path.append(SRC)

from models.nebula_model import NebulaTiny
from tokenization.tokenizer import tokenize, tokens_to_ids

def load_checkpoint(ckpt_path, vocab_path):
    with open(vocab_path, "r") as vf:
        vocab = json.load(vf)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = NebulaTiny(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"], nhead=cfg["heads"],
                       num_layers=cfg["layers"], dim_feedforward=cfg["ff"],
                       max_len=cfg["max_len"], num_classes=2, chunk_size=cfg["chunk_size"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, vocab, cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocab_file", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--sample_index", type=int, default=0)
    args = ap.parse_args()

    model, vocab, cfg = load_checkpoint(args.checkpoint, args.vocab_file)

    rows = []
    with open(args.data_file, "r") as f:
        for line in f:
            i, y, t = line.rstrip("\\n").split("\\t", 2)
            rows.append((int(i), int(y), t))
    i, y, t = rows[args.sample_index]
    ids = tokens_to_ids(tokenize(t), vocab, cfg["max_len"])
    x = torch.tensor([ids], dtype=torch.long)

    x_emb = model.embed(x).detach().requires_grad_(True)
    h = model.pos(x_emb)
    for layer in model.layers:
        h = layer(h, (x != 0).unsqueeze(1).unsqueeze(2))
    pooled = h[:,0,:]
    logits = model.cls(pooled)
    prob = torch.softmax(logits, dim=-1)[0,1]
    prob.backward()
    grads = x_emb.grad.abs().sum(dim=-1).squeeze(0)
    id2tok = {v:k for k,v in vocab.items()}
    tokens = [id2tok.get(tok_id, "<unk>") for tok_id in x[0].tolist()]

    pairs = [(tok, float(grad)) for tok, grad in zip(tokens, grads.tolist()) if tok != "<pad>"]
    pairs.sort(key=lambda x: x[1], reverse=True)
    print("Sample index:", i, "| True label:", y)
    print("Top tokens by gradient importance:")
    for tok, score in pairs[:15]:
        print(f"{tok}\t{score:.5f}")

if __name__ == "__main__":
    main()
