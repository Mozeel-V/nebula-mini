echo "Running Window Evaluation"
python src/eval/window_eval.py \
  --data_file data/processed/dataset.txt \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --out_dir results/window_eval \
  --window_unit event \
  --events_per_window 32 \
  --stride_events 8 \
  --sample_limit 200
