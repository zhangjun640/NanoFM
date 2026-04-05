# NanoFM1 README

## Overview

This project follows a **site-level** prediction workflow. The recommended pipeline is:

1. **Read-level feature extraction** (`extra_feature/extra.py`)
2. **Site-level aggregation** (`agress/agress.py`)
3. **Sequence embedding extraction** (`embedding/embedding.py`)
4. **Model training** (`scripts/train_site.py`)
5. **Inference and evaluation** (`scripts/predict_site.py`)

The key update in this README is that **aggregation must be inserted between read-level feature extraction and embedding extraction**.

---

## Recommended pipeline

### Step 1. Extract read-level features

Convert raw FAST5 files and aligned SAM records into **read-level 5-mer features**.

```bash
python NanoFM1/extra_feature/extra.py     --fast5 /path/to/fast5_dir     --sam /path/to/aligned.sam     --reference /path/to/reference.fasta     --motif DRACH     --output ./features.read_level.tsv     --process 8
```

Output: `features.read_level.tsv`

This file contains per-read records for candidate 5-mers, including:
- read ID
- chromosome / transcript ID
- position
- 5-mer sequence
- signal summary statistics
- base quality
- processed signals
- mismatch / insertion / deletion indicators

---

### Step 2. Aggregate read-level records into site-level records

After read-level extraction, run **aggress/aggregation** before embedding.

```bash
python NanoFM1/agress/agress.py     --input ./features.read_level.tsv     --output ./features.site_level.tsv     --min_cov 20
```

Output: `features.site_level.tsv`

This step groups multiple reads from the same site into one site-level entry. Each line should correspond to one site and should contain:
- `Chrom`
- `Pos`
- `Kmer`
- aggregated read block
- optional `Label`

This is the format expected by `train_site.py`, `predict_site.py`, and `MyDataSet_site.py`.

---

### Step 3. Extract sequence embeddings from site-level candidates

Use the pre-trained structRFM model to compute 5-mer embeddings **after aggregation**.

```bash
python NanoFM1/embedding/embedding.py     --model_path /path/to/structRFM_checkpoint     --input ./features.site_level.tsv     --reference_fasta /path/to/reference.fasta     --output ./embeddings.parquet
```

Output: `embeddings.parquet`

The embedding file is keyed by genomic / transcript site and is later loaded during training and inference.

---

### Step 4. Train the site-level model

```bash
python NanoFM1/scripts/train_site.py     --train_file ./train_features.site_level.tsv     --valid_file ./val_features.site_level.tsv     --embedding_file ./embeddings.parquet     --save_dir ./model_checkpoints     --model_type comb     --epochs 100     --batch_size 64     --lr 1e-4     --use_cross_attention True
```

Supported `--model_type` values:
- `basecall`: handcrafted / basecall-derived features only
- `raw_signals`: raw current signal only
- `comb`: joint model with feature fusion

Notes:
- Model checkpoints are saved automatically for each epoch.
- Early stopping is triggered according to validation accuracy.

---

### Step 5. Predict on unseen sites

```bash
python NanoFM1/scripts/predict_site.py     --model ./model_checkpoints/checkpoint.pt     --input_file ./test_features.site_level.tsv     --embedding_file ./embeddings.parquet     --output_file ./results/predictions.tsv     --model_type comb     --batch_size 64     --nproc 4
```

Prediction output columns:
- `Chrom`
- `Pos`
- `Kmer`
- `Probability`
- `PredLabel`
- `TrueLabel`

After prediction, a metrics file such as `predictions_metrics.csv` is generated automatically.

Metrics may include:
- Accuracy
- ROC AUC
- PR AUC
- F1-score
- Sensitivity / Recall
- Specificity
- Precision

---

## Why aggregation must come before embedding

`extra.py` produces **read-level** records, while the downstream training and prediction scripts operate on **site-level** aggregated records. Therefore, the correct data flow is:

```text
FAST5 + SAM
  -> extra.py
  -> read-level TSV
  -> agress.py
  -> site-level TSV
  -> embedding.py
  -> embeddings.parquet
  -> train_site.py / predict_site.py
```

This ordering keeps the embedding stage aligned with the same site definition used by the site-level dataset and model.

---

## Core arguments

- `--model_type`: choose `basecall`, `raw_signals`, or `comb`
- `--min_cov`: number of reads sampled per site, default `20`
- `--signal_lens`: signal length per base, default `65`
- `--use_cross_attention`: whether to enable bidirectional cross-attention fusion
- `--nproc`: number of prediction processes

---

## Suggested directory structure

```text
NanoFM1/
├── README.md
├── extra_feature/
│   └── extra.py
├── agress/
│   ├── agress.py
│   └── README.md
├── embedding/
│   └── embedding.py
└── scripts/
    ├── train_site.py
    └── predict_site.py

```

---

## Quick start

```bash
# 1) read-level feature extraction
python NanoFM1/extra_feature/extra.py     --fast5 /data/fast5     --sam /data/aligned.sam     --reference /data/reference.fa     --motif DRACH     --output ./features.read_level.tsv     --process 8

# 2) site-level aggregation
python NanoFM1/agress/agress.py     --input ./features.read_level.tsv     --output ./features.site_level.tsv     --min_cov 20

# 3) embedding extraction
python NanoFM1/embedding/embedding.py     --model_path /models/structRFM.ckpt     --input ./features.site_level.tsv     --reference_fasta /data/reference.fa     --output ./embeddings.parquet

# 4) model training
python NanoFM1/scripts/train_site.py     --train_file ./train_features.site_level.tsv     --valid_file ./val_features.site_level.tsv     --embedding_file ./embeddings.parquet     --save_dir ./model_checkpoints     --model_type comb     --epochs 100     --batch_size 64     --lr 1e-4     --use_cross_attention True

# 5) inference
python NanoFM1/scripts/predict_site.py     --model ./model_checkpoints/checkpoint.pt     --input_file ./test_features.site_level.tsv     --embedding_file ./embeddings.parquet     --output_file ./results/predictions.tsv     --model_type comb     --batch_size 64     --nproc 4
```
