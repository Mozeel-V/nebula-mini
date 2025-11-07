python real_data/saliency_attack_sampler.py \
  --dataset real_data/processed/dataset_balanced.tsv \
  --saliency_json real_data/results/saliency_positions.json \
  --api_effect real_data/checkpoints/api_effectiveness.json \
  --api_pool real_data/checkpoints/api_candidates.json \
  --out real_data/results/attacks/saliency_sampler_adversarial_small.tsv \
  --n_replace 6 \
  --k 32 \
  --sample_limit 50
