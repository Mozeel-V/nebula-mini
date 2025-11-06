echo "Running Window Evaluation"
python src/eval/window_eval.py \
  --data_file data/processed/adversarial_hotflip.txt \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --window_unit event --events_per_window 16 --stride_events 4 \
  --out_dir results/window_eval_hotflip \

