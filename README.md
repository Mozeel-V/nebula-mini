# 🌌 Nebula Mini – Adversarial Malware Detection & Robustness Analysis

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-3670A0?logo=python&logoColor=white)](https://www.python.org/)
[![Trained using PyTorch](https://img.shields.io/badge/Trained%20using-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/Build-Active-blue)](https://github.com/Mozeel-V/nebula-mini)

A compact, research-oriented, end-to-end reimplementation inspired by the [Nebula](https://arxiv.org/abs/2310.10664) pipeline for dynamic malware analysis using Transformer architectures. Extended with adversarial attack generation, saliency-based interpretability and robustness evaluation.

It uses:
- JSON sandbox-like logs → domain normalization → text flattening
- Simple whitespace tokenization (no external BPE dependency)
- Tiny Transformer encoder (no chunked attention)
- Binary detection demo (malicious vs benign) on a **toy dataset** that runs without a GPU
- Insert, Genetic Algorithm (GA), and Saliency-guided adversarial attacks and retraining to test detector robustness
- Visualization of attack success metrics
- Automatic checkpointing, metric logging, and TorchScript export
- Window-based evaluation with per-sample temporal plots and auto threshold tuning
- Propagation risk analysis for repeated benign windows

> Goal: Build a minimal, fully working malware behavior model pipeline that runs locally and supports adversarial ML experimentation and explainability. 

## 🧱 How To Run

Step 1: Clone the Repository

```bash
git clone https://github.com/Mozeel-V/nebula-mini.git
cd nebula-mini
```

Step 2: Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

Step 3: Generate long synthetic traces

```bash
python src/preprocess/synth_generator.py --n_samples 200 --n_events 200
```

Step 4: Preprocess raw JSON logs to processed text
```bash
python src/preprocess/to_text.py --raw_dir data/raw --out_file data/processed/dataset.txt --manifest data/manifests/splits.json
```

Step 5: Train tiny baseline detector model, might overfit on small data
```bash
python src/train/train_supervised.py   --data_file data/processed/dataset.txt   --splits data/manifests/splits.json   --max_len 512 --d_model 128 --heads 4 --layers 1 --epochs 6 --batch_size 8 --chunk_size 0  
```

During training the following files will be generated:

- `checkpoints/last.pt` — latest model for resuming
- `checkpoints/best.pt` — best validation model
- `checkpoints/full_model.pkl` — pickled full model
- `checkpoints/model_scripted.pt` — TorchScript version
- `checkpoints/last_metrics.json`, `best_metrics.json`, `test_metrics.json` — performance logs

Step 6: Run simple explain on 1 sample
```bash
python src/eval/explain.py --checkpoint checkpoints/best.pt --vocab_file checkpoints/vocab.json --sample_index 0 --data_file data/processed/dataset.txt
```

Step 7: Run simple adversarial attack
```bash
bash scripts/run_attack_simple.sh
```
→ generates `data/processed/adversarial.txt` (benign-padded malware traces)

Step 8: Full evaluation suite (Insert, GA, Saliency attacks)
```bash
bash scripts/run_evals.sh
```

This runs:

- `api_candidates.py` → builds benign API pool
- `eval_attacks.py` → Insert-based attacks
- `search_attack.py` → GA/Hill-Climb attacks
- `saliency_selector.py` + `saliency_attack.py` → saliency-guided mutations

Generates `results/*.json` and final bar chart figures\

Step 9: Retrain with adversarial data
```bash
bash scripts/run_retrain.sh
```
→ merges adversarial + original data and fine-tunes the model


### 🧩 Window-level Supervision and MIL Evaluation

To capture temporal behavior within traces, the pipeline was extended to include:
- **Rule-based window labeling** (`make_window_dataset.py`) for pseudo-supervised learning.
- **Window-level fine-tuning** (`train_window_supervised.py`) to identify malicious event bursts.
- **Multiple Instance Learning (MIL)** (`train_mil.py`) to align with sample-level detection (a sample is malicious if any window is malicious).
- **Time-series visualization** (`window_eval.py`) showing per-window malware probabilities before and after attacks.
- Results and plots are saved in `results/figures/`, enabling temporal pattern inspection.

These experiments aim to improve interpretability and robustness by linking trace dynamics to detection outcomes.

Example Result:
```bash
[INFO] Loaded 200 samples
[DEBUG] Malware max window scores min/med/max/mean = (0.8449975252151489, 0.8462614417076111, 0.8501846790313721, 0.8462317681312561)
[DEBUG] Benign  max window scores min/med/max/mean = (0.8431179523468018, 0.8450426459312439, 0.8469246029853821, 0.8449537390470505)
[INFO] Auto-picked threshold = 0.846  (best F1=0.813, tp=89, fp=30, fn=11)
[RESULT] Sample-level summary: {'threshold': 0.8455071449279785, 'accuracy': 0.795, 'precision': 0.7478991596638593, 'recall': 0.8899999999999911, 'f1': 0.8127853881273501, 'tp': 89, 'fp': 30, 'tn': 70, 'fn': 11}
```


## 📁 Directory Structure
```bash
Nebula_Mini/
│
├── .gitignore                 # Specifies files and directories that should be ignored by Git
├── README.md                  # Project description, installation, usage, and details
├── LICENSE                    # Open-source MIT license for the project
│
├── data/                       # Directory containing data files 
│   ├── manifests/splits.json   # train/val/test split indices
│   ├── processed/              # for adversarial samples
│   └── raw/                    # toy dataset (JSON) with `malware_*.json` and `benign_*.json`
│
├── src/                        # Source files
│   ├── preprocess/synth_generator.py   # to generate long traces of activity
│   ├── preprocess/field_filters.py     # which JSON fields we keep 
│   ├── preprocess/normalizers.py       # IP/hash/domain/path normalization   
│   ├── preprocess/to_text.py           # flattens JSON logs to text, saves splits   
│   ├── tokenization/tokenizer.py       # whitespace tokenizer + vocab builder  
│   ├── models/chunked_attention.py     # optional local attention 
│   ├── models/nebula_model.py          # tiny Transformer encoder + classifier 
│   ├── train/train_supervised.py       # training loop (binary)
│   ├── attacks/simple_attacks.py       # insertion and replacement attacks
│   ├── attacks/ga_attack.py            # hill-climb / GA attacks 
│   ├── attacks/saliency_selector.py    # gradient-based importance
│   ├── attacks/saliency_attack.py      # saliency-biased adversarial gen
│   ├── eval/plot_results.py,plot.py    # for plotting the metrics
│   ├── eval/metrics.py                 # TPR@FPR, AUC, F1                
│   ├── eval/window_eval.py             # window-level evaluation, propagation, plots
│   └── eval/explain.py                 # prints top tokens by attention & gradients (simple)
│
├── scripts/                    # Directory containing bash scripts 
│   ├── run_attack_simple.sh    # to run the adversarial attacks
│   ├── run_evals.sh            # to compute and plot evaluation metrics
│   └── run_retrain.sh          # to retrain the model after the attacks
│
├── checkpoints/                # all model + log artifacts  
├── figures/                    # ASR bar and ROC plots
├── requirements.txt            # required python packages
└── setup.py                    # enables `pip install -e .`
```

## 📊 Output Example

After a successful training run:
```bash
Epoch 1 | train_loss=0.7179 | val_auc=0.460 | val_f1=0.400 | TPR@1e-3=0.100
Epoch 2 | train_loss=0.6439 | val_auc=0.595 | val_f1=0.480 | TPR@1e-3=0.200
Epoch 3 | train_loss=0.5414 | val_auc=0.605 | val_f1=0.516 | TPR@1e-3=0.200
Epoch 4 | train_loss=0.4999 | val_auc=0.630 | val_f1=0.400 | TPR@1e-3=0.300
Epoch 5 | train_loss=0.4630 | val_auc=0.595 | val_f1=0.400 | TPR@1e-3=0.400
Epoch 6 | train_loss=0.4222 | val_auc=0.730 | val_f1=0.500 | TPR@1e-3=0.300
TEST: {'auc': 0.8044444444444444, 'f1': 0.7333333333333333, 'tpr_at_1e-3': 0.4, 'tpr_at_1e-4': 0.4}
Saved to checkpoints/best.pt
```

## 🤝 Contributions

Feel free to fork, raise issues, or submit PRs to improve this project!

## 👤 Author

Mozeel Vanwani

Computer Science and Engineering Undergrad 💻

Indian Institute of Technology Kharagpur 🎓

Email 📧: vanwani.mozeel@gmail.com


