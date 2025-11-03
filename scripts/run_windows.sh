python src/preprocess/make_window_dataset.py \
  --data_file data/processed/dataset.txt \
  --out_file data/processed/windows.tsv \
  --events_per_window 16 --stride_events 4

python src/train/train_window_supervised.py \
  --windows_tsv data/processed/windows.tsv \
  --ckpt checkpoints/best.pt \
  --vocab checkpoints/vocab.json \
  --epochs 3 --batch_size 128 --lr 3e-5 \
  --out_ckpt checkpoints/best_window_supervised.pt

python src/train/train_mil.py \
  --data_file data/processed/dataset.txt \
  --splits data/manifests/splits.json \
  --ckpt checkpoints/best_window_supervised.pt \
  --vocab checkpoints/vocab.json \
  --unit event --events_per_window 16 --stride_events 4 \
  --epochs 3 --batch_size 8 --lr 2e-5 \
  --out_dir checkpoints

python src/eval/window_eval.py \
  --data_file data/processed/dataset.txt \
  --ckpt checkpoints/best_mil.pt \
  --vocab checkpoints/vocab.json \
  --window_unit event --events_per_window 16 --stride_events 4 \
  --out_dir results/window_eval_after_mil
