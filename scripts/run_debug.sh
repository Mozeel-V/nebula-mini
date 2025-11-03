python src/eval/inspect_windows.py \
  --data_file data/processed/dataset.txt \
  --vocab_file checkpoints/vocab.json \
  --window_unit token --tokens_per_window 512 --stride_tokens 128 --max_len 512 \
  --sample_limit 100

python src/eval/debug_logits.py \
  --data_file data/processed/dataset.txt \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --sample_id 132 \
  --tokens_per_window 512 --stride_tokens 128
