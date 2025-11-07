python src/attacks/saliency_selector.py \
  --data_file real_data/processed/dataset_balanced.tsv \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --out real_data/results/saliency_positions.json \
  --k 64 \
  --sample_limit 500
