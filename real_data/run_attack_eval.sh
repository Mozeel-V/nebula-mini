#!/usr/bin/env bash
set -euo pipefail
ROOT=$(pwd)
echo "[INFO] Running attack + eval pipeline (adjust paths/flags as needed)"

# 0) produce balanced dataset if not already
# python3 real-data/process_malware_traces.py --input_dir Malware-Traces --out real-data/processed/dataset_raw.tsv
# python3 real-data/balance_dataset.py --input real-data/processed/dataset_raw.tsv --output real-data/processed/dataset_balanced.tsv

# 1) Evaluate baseline (before attack)
python3 src/eval/window_eval_plot.py \
  --data_file real-data/processed/dataset_balanced.tsv \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --out_dir real-data/results/window_eval_plot_before \
  --window_unit event --events_per_window 16 --stride_events 4

# 2) select zero-FP threshold
python3 real-data/select_zero_fp.py --window_eval real-data/results/window_eval_plot_before/window_eval.json --out zero_fp_threshold.json

# 3) Run attack to produce adversarial dataset
# Example with GA attack; adapt args to ga_attack.py interface in repo
# python3 src/attacks/ga_attack.py \
  --dataset real-data/processed/dataset_balanced.tsv \
  --out_dir real-data/results/attacks/ga \
  --budget 1000 \
  --population 50 \
  --generations 200

# assume ga_attack writes results/attacks/ga/dataset_adversarial.txt
# 4) Evaluate attacked dataset (after)
python3 src/eval/window_eval_plot.py \
  --data_file real-data/results/attacks/ga/dataset_adversarial.txt \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --out_dir real-data/results/window_eval_plot_after \
  --window_unit event --events_per_window 16 --stride_events 4

# 5) ensure we use the zero-FP threshold on both before & after
python3 real-data/apply_threshold.py --window_eval real-data/results/window_eval_plot_before/window_eval.json --threshold $(jq -r .threshold zero_fp_threshold.json) --out real-data/results/window_eval_plot_before/window_eval_fixed_before.json
python3 real-data/apply_threshold.py --window_eval real-data/results/window_eval_plot_after/window_eval.json --threshold $(jq -r .threshold zero_fp_threshold.json) --out real-data/results/window_eval_plot_after/window_eval_fixed_after.json

# 6) Compare & produce side-by-side plots
python3 real-data/compare_before_after.py --before real-data/results/window_eval_plot/window_eval.json --after real-data/results/window_eval_plot_after/window_eval.json --zero_fp zero_fp_threshold.json --out_dir real-data/results/comparisons --only_malware

echo "[DONE] pipeline finished; check real-data/results/comparisons"
