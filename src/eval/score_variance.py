import argparse, json, numpy as np
data = json.load(open("results/window_eval/window_eval.json"))
stds = []
for sid, s in data["samples"].items():
    if s["scores"]:
        stds.append(float(np.std(s["scores"])))
print("per-sample score std: min/med/max =", min(stds), np.median(stds), max(stds))