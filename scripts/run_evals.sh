#!/usr/bin/env bash
set -e

# Config
DATA_FILE="data/processed/dataset.txt"
VOCAB="checkpoints/vocab.json"
CKPT="checkpoints/best.pt"
API_JSON="checkpoints/api_candidates.json"

RESULTS_DIR="results"
FIG_DIR="figures"
mkdir -p "$RESULTS_DIR" "$FIG_DIR"

echo "[1/5] Building API candidates..."
python src/attacks/api_candidates.py --raw_dir data/raw --out "$API_JSON" --top_k 50

echo "[2/5] Running INSERT attack..."
python src/attacks/eval_attacks.py \
  --data_file "$DATA_FILE" \
  --ckpt "$CKPT" \
  --vocab "$VOCAB" \
  --api_candidates "$API_JSON" \
  --out "$RESULTS_DIR/insert_results.json" \
  --insert_n 100 \
  --sample_limit 200

echo "[3/5] Running GA/Hill-Climb attack..."
python src/attacks/search_attack.py \
  --data_file "$DATA_FILE" \
  --ckpt "$CKPT" \
  --vocab "$VOCAB" \
  --api_candidates "$API_JSON" \
  --out "$RESULTS_DIR/ga_results.json" \
  --iters 25 \
  --sample_limit 200

echo "[4/5] Running Saliency selector + attack..."
python src/attacks/saliency_selector.py \
  --data_file "$DATA_FILE" \
  --ckpt "$CKPT" \
  --vocab "$VOCAB" \
  --out "$RESULTS_DIR/saliency_positions.json" \
  --k 32 \
  --sample_limit 200

python src/attacks/saliency_attack.py \
  --data_file "$DATA_FILE" \
  --ckpt "$CKPT" \
  --vocab "$VOCAB" \
  --api_candidates "$API_JSON" \
  --saliency_json "$RESULTS_DIR/saliency_positions.json" \
  --out "$RESULTS_DIR/saliency_results.json" \
  --n_replace 8 \
  --sample_limit 200

echo "[5/5] Plotting ASR bar..."
python src/eval/plot_results.py \
  --inputs "$RESULTS_DIR/insert_results.json" "$RESULTS_DIR/ga_results.json" "$RESULTS_DIR/saliency_results.json" \
  --labels Insert GA Saliency \
  --out "$FIG_DIR/asr_bar.png"

echo "Done. Figures in $FIG_DIR, results in $RESULTS_DIR"
