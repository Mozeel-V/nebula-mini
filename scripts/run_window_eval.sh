echo "Running Window Evaluation"
python src/eval/window_eval.py \
  --data_file data/processed/dataset.txt \
  --ckpt checkpoints/best_windows.pt \
  --vocab checkpoints/vocab.json \
  --window_unit event --events_per_window 16 --stride_events 8 \
  --out_dir results/window_eval_fine_tuned

