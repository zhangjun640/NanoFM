# NanoFM

## Overview

NanoFM follows a **site-level RNA modification prediction** workflow for nanopore direct RNA sequencing. The recommended pipeline is:

1. **Read-level feature extraction** (`extra_feature/extra.py`)
2. **Site-level aggregation** (`agress/agress.py`)
3. **Sequence embedding extraction** (`embedding/embedding.py`)
4. **Model training** (`scripts/train_site.py`)
5. **Inference and evaluation** (`scripts/predict_site.py`)

A key point in this workflow is that **site-level aggregation must be inserted between read-level feature extraction and sequence embedding extraction**.

---

## Installation

### 1. Clone NanoFM

```bash
git clone https://github.com/<your-username>/NanoFM.git
cd NanoFM
```

### 2. Create a Python environment

It is recommended to use a clean conda environment:

```bash
conda create -n nanofm python=3.10 -y
conda activate nanofm
```

Then install the core Python dependencies required by NanoFM:

```bash
pip install numpy pandas scipy scikit-learn tqdm pyarrow h5py statsmodels
pip install torch torchvision torchaudio
```

If you use GPU training or inference, install the PyTorch build that matches your CUDA version.

---

## structRFM environment and installation

NanoFM uses **structRFM** to extract sequence embeddings for candidate 5-mer sites.

structRFM repository:

```text
https://github.com/heqin-zhu/structRFM
```

You can install and use structRFM in **two ways**.

### Option A. Lightweight inference setup (recommended for NanoFM embedding extraction)

According to the structRFM project, a lightweight inference workflow can be prepared by installing `transformers`, `structRFM`, and `BPfold`, then downloading the released pretrained checkpoint and setting the `structRFM_checkpoint` environment variable.

```bash
pip install transformers structRFM BPfold
wget https://github.com/heqin-zhu/structRFM/releases/latest/download/structRFM_checkpoint.tar.gz
tar -xzf structRFM_checkpoint.tar.gz
export structRFM_checkpoint=/absolute/path/to/structRFM_checkpoint
```

To make the environment variable persistent, add it to your shell profile such as `~/.bashrc`:

```bash
echo 'export structRFM_checkpoint=/absolute/path/to/structRFM_checkpoint' >> ~/.bashrc
source ~/.bashrc
```

### Option B. Full structRFM repository environment

If you also want to run structRFM directly, reproduce its original environment, or modify its code, use the official repository environment:

```bash
git clone https://github.com/heqin-zhu/structRFM.git
cd structRFM
conda env create -f structRFM_environment.yaml
conda activate structRFM
```

Then return to NanoFM:

```bash
cd /path/to/NanoFM
```

### Quick check for structRFM

If structRFM is installed correctly, the following command should run without error:

```bash
python -c "from transformers import AutoModel, AutoTokenizer; model = AutoModel.from_pretrained('heqin-zhu/structRFM'); tokenizer = AutoTokenizer.from_pretrained('heqin-zhu/structRFM'); print('structRFM is ready')"
```

If your embedding script is written to read the local released checkpoint, make sure `structRFM_checkpoint` points to the decompressed checkpoint directory.

---

## Recommended pipeline

### Step 1. Extract read-level features

Convert raw FAST5 files and aligned SAM records into **read-level 5-mer features**.

```bash
python NanoFM1/extra_feature/extra.py \
    --fast5 /path/to/fast5_dir \
    --sam /path/to/aligned.sam \
    --reference /path/to/reference.fasta \
    --motif DRACH \
    --output ./features.read_level.tsv \
    --process 8
```

Output:

```text
features.read_level.tsv
```

This file contains per-read records for candidate 5-mers, including:
- read ID
- chromosome or transcript ID
- position
- 5-mer sequence
- signal summary statistics
- base quality
- processed signals
- mismatch / insertion / deletion indicators

---

### Step 2. Aggregate read-level records into site-level records

After read-level extraction, run **agress** before embedding.

```bash
python NanoFM1/agress/agress.py \
    --input ./features.read_level.tsv \
    --output ./features.site_level.tsv \
    --min_cov 20
```

Output:

```text
features.site_level.tsv
```

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
python NanoFM1/embedding/embedding.py \
    --model_path /path/to/structRFM_checkpoint \
    --input ./features.site_level.tsv \
    --reference_fasta /path/to/reference.fasta \
    --output ./embeddings.parquet
```

Output:

```text
embeddings.parquet
```

The embedding file is keyed by genomic or transcript site and is loaded during training and inference.

---

### Step 4. Train the site-level model

```bash
python NanoFM1/scripts/train_site.py \
    --train_file ./train_features.site_level.tsv \
    --valid_file ./val_features.site_level.tsv \
    --embedding_file ./embeddings.parquet \
    --save_dir ./model_checkpoints \
    --model_type comb \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --use_cross_attention True
```

Supported `--model_type` values:
- `basecall`: handcrafted or basecall-derived features only
- `raw_signals`: raw current signal only
- `comb`: joint model with feature fusion

Notes:
- Model checkpoints are saved automatically for each epoch.
- Early stopping is triggered according to validation accuracy.

---

### Step 5. Predict on unseen sites

```bash
python NanoFM1/scripts/predict_site.py \
    --model ./model_checkpoints/checkpoint.pt \
    --input_file ./test_features.site_level.tsv \
    --embedding_file ./embeddings.parquet \
    --output_file ./results/predictions.tsv \
    --model_type comb \
    --batch_size 64 \
    --nproc 4
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
python NanoFM1/extra_feature/extra.py \
    --fast5 /data/fast5 \
    --sam /data/aligned.sam \
    --reference /data/reference.fa \
    --motif DRACH \
    --output ./features.read_level.tsv \
    --process 8

# 2) site-level aggregation
python NanoFM1/agress/agress.py \
    --input ./features.read_level.tsv \
    --output ./features.site_level.tsv \
    --min_cov 20

# 3) embedding extraction
python NanoFM1/embedding/embedding.py \
    --model_path /models/structRFM_checkpoint \
    --input ./features.site_level.tsv \
    --reference_fasta /data/reference.fa \
    --output ./embeddings.parquet

# 4) model training
python NanoFM1/scripts/train_site.py \
    --train_file ./train_features.site_level.tsv \
    --valid_file ./val_features.site_level.tsv \
    --embedding_file ./embeddings.parquet \
    --save_dir ./model_checkpoints \
    --model_type comb \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-4 \
    --use_cross_attention True

# 5) inference
python NanoFM1/scripts/predict_site.py \
    --model ./model_checkpoints/checkpoint.pt \
    --input_file ./test_features.site_level.tsv \
    --embedding_file ./embeddings.parquet \
    --output_file ./results/predictions.tsv \
    --model_type comb \
    --batch_size 64 \
    --nproc 4
```

---

## Citation

If you use structRFM in this workflow, please also cite the structRFM project and paper.
