python src/eval/window_eval_plot.py \
  --data_file data/processed/dataset.txt \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --out_dir results/window_eval_plot \
  --window_unit event --events_per_window 16 --stride_events 4
