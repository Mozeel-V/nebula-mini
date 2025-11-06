python src/attacks/hotflip_minimax.py \
  --data_file data/processed/dataset.txt \
  --out_file data/processed/adversarial_hotflip.txt \
  --ckpt checkpoints/best_windows.pt \
  --vocab checkpoints/vocab.json \
  --token_stats results/token_stats.json \
  --malware_index auto \
  --max_flips 3
