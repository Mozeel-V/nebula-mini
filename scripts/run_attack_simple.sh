#!/usr/bin/env bash
python src/attacks/simple_attacks.py --data_file data/processed/dataset.txt --out_file data/processed/adversarial.txt --strategy insert --insert_n 100 --sample_limit 50
