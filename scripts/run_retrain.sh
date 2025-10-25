#!/usr/bin/env bash
python - <<'PY'
from pathlib import Path
p=Path('data/processed/dataset.txt')
adv=Path('data/processed/adversarial.txt')
out=Path('data/processed/dataset_adversarial.txt')
if not adv.exists(): print('adv missing'); raise SystemExit(1)
lines=p.read_text().splitlines()
adv_lines=adv.read_text().splitlines()
out.write_text('\n'.join(lines+adv_lines))
print('wrote', out)
PY
python src/train/train_supervised.py --data_file data/processed/dataset_adversarial.txt --splits data/manifests/splits.json --max_len 512 --d_model 128 --epochs 6 --batch_size 8
