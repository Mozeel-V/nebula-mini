import json, torch, numpy as np
from pathlib import Path
from tokenization.tokenizer import tokenize, tokens_to_ids
from models.nebula_model import NebulaTiny

ck = torch.load("checkpoints/best.pt", map_location="cpu")
cfg = ck["config"]; state = ck.get("model", ck.get("model_state"))
vocab = json.load(open("checkpoints/vocab.json"))
model = NebulaTiny(cfg["vocab_size"], cfg["d_model"], cfg["heads"], cfg["layers"],
                   cfg["ff"], cfg["max_len"], num_classes=2, chunk_size=cfg.get("chunk_size",0))
model.load_state_dict(state); model.eval()

mal_means = []; ben_means = []
with open("data/processed/dataset.txt") as f:
    for line in f:
        i,y,t = line.rstrip("\n").split("\t",2)
        ids = tokens_to_ids(tokenize(t), vocab, max_len=cfg["max_len"])
        x = torch.tensor([ids])
        with torch.no_grad():
            lo = model(x)[0].numpy()
        pr = np.exp(lo - lo.max()) / np.exp(lo - lo.max()).sum()
        (mal_means if int(y)==1 else ben_means).append(pr)
mal_mean = np.mean(np.array(mal_means), axis=0)
ben_mean = np.mean(np.array(ben_means), axis=0)
print("mean probs on malware:", mal_mean, "benign:", ben_mean)
print("=> malware class index is", int(np.argmax(mal_mean)))
