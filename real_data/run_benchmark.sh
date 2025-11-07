python real_data/benchmark_window_scoring.py \
  --dataset real_data/processed/dataset_balanced.tsv \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --n 50 \
  --window_unit event \
  --events_per_window 16 \
  --stride_events 8 \
  --batch_size 128
