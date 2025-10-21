# Nebula Mini Implementation

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3670A0?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Build-Active-blue)](https://github.com/Mozeel-V/nebula-mini)

A tiny, end-to-end reimplementation in spirit of the [Nebula](https://arxiv.org/abs/2310.10664) pipeline for dynamic malware logs.
It uses:
- JSON sandbox-like logs → domain normalization → text flattening
- Simple whitespace tokenization (no external BPE dependency)
- Tiny Transformer encoder (no chunked attention)
- Binary detection demo (malicious vs benign) on a **toy dataset** that runs without a GPU

> Goal: **To Prove that the pipeline works** on local machine with minimal dependencies. 

## How To Run

Step 1: Clone the Repository

```bash
git clone https://github.com/Mozeel-V/nebula-mini.git
```

Step 2: Create a virtual environment and install dependencies

```bash
python3 -m venv .venv && source .venv/bin/activate  
pip install -r requirements.txt
```

Step 3: Preprocess raw JSON logs to processed text
```bash
python src/preprocess/to_text.py --raw_dir data/raw --out_file data/processed/dataset.txt --manifest data/manifests/splits.json
```

Step 4: Train tiny model, might overfit on small data
```bash
python src/train/train_supervised.py   --data_file data/processed/dataset.txt   --splits data/manifests/splits.json   --max_len 256 --d_model 128 --heads 4 --layers 1 --epochs 6 --batch_size 16   --chunk_size 0  
```

Step 5: Run simple explain on 1 sample
```bash
python src/eval/explain.py --checkpoint checkpoints/best.pt --vocab_file checkpoints/vocab.json --sample_index 0 --data_file data/processed/dataset.txt
```

## Directory Structure
```bash
Nebula_Mini/
│
├── .gitignore                 # Specifies files and directories that should be ignored by Git
├── README.md                  # Project description, installation, usage, and details
│
├── data/                       # Directory containing data files 
│   ├── manifests/splits.json   # train/val/test split indices
│   └── raw                     # toy dataset (JSON) with `malware_*.json` and `benign_*.json`
│
├── src/                        # Source files
│   ├── preprocess/field_filters.py     # which JSON fields we keep 
│   ├── preprocess/normalizers.py       # IP/hash/domain/path normalization   
│   ├── preprocess/to_text.py           # flattens JSON logs to text, saves splits   
│   ├── tokenization/tokenizer.py       # whitespace tokenizer + vocab builder  
│   ├── models/chunked_attention.py     # optional local attention 
│   ├── models/nebula_model.py          # tiny Transformer encoder + classifier 
│   ├── train/train_supervised.py       # training loop (binary)
│   ├── eval/metrics.py                 # TPR@FPR, AUC, F1                
│   └── eval/explain.py                 # prints top tokens by attention & gradients (simple)
│ 
└── LICENSE                             # Open-source MIT license for the project
```


## Notes

- This is a demo scaffold, not the exact paper code.
- Swap in your own logs at `data/raw/` and rerun preprocessing.

## Contributions

Feel free to fork, raise issues, or submit PRs to improve this project!

## Author

Mozeel Vanwani | IIT Kharagpur CSE

Email: [vanwani.mozeel@gmail.com]


