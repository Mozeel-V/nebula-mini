python real_data/compute_effectiveness.py \
  --dataset real_data/processed/dataset_balanced.tsv \
  --api_pool real_data/checkpoints/api_candidates.json \
  --saliency_json real_data/results/saliency_positions.json \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --out real_data/checkpoints/api_effectiveness.json \
  --sample_limit 200 \
  --k 32 \
  --per_api_positions 2
