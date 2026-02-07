# Kaggle Setup Guide - Medical MTL Project

**Date**: 2026-02-07
**Goal**: Deploy your research project to Kaggle and run experiments on free T4 GPU

---

## Overview

This guide shows you how to:
1. Push your code to GitHub
2. Pull it into Kaggle notebooks
3. Handle datasets efficiently
4. Run experiments on free T4 GPU

---

## Part 1: Push Code to GitHub

### Step 1: Initialize Git Repository

```bash
# In your project directory
cd h:\Projects\Research\Cross_Talk_Medical_LLM

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Medical MTL project

- 73 Python modules implemented
- 22 YAML configs for experiments
- 11 execution scripts
- Full setup with medical-mtl environment
- Ready for Kaggle deployment"
```

### Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `medical-cross-task-transfer` (or your choice)
3. Description: "Cross-Task Knowledge Transfer with Small Language Models for Medical NLP"
4. **Privacy**: Choose **Private** (recommended for research)
5. **Do NOT** initialize with README (you already have one)
6. Click "Create repository"

### Step 3: Link Local Repo to GitHub

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/medical-cross-task-transfer.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 4: Verify .gitignore is Working

Check that sensitive files are NOT pushed:

```bash
# This should NOT show .env file
git status

# If .env appears, remove it:
git rm --cached .env
git commit -m "Remove .env from tracking"
git push
```

**CRITICAL**: Your `.gitignore` already excludes:
- `.env` (contains HF_TOKEN and WANDB_API_KEY)
- `data/raw/` (datasets)
- `checkpoints/` (model checkpoints)
- `__pycache__/` (Python cache)

---

## Part 2: Pull Code into Kaggle

### Method 1: Clone from GitHub (Recommended)

1. **Create new Kaggle Notebook**:
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"
   - Settings → Accelerator → **GPU T4 x2** (free)
   - Settings → Language → **Python**

2. **Clone your repository**:

```python
# In first cell of Kaggle notebook
!git clone https://github.com/YOUR_USERNAME/medical-cross-task-transfer.git
%cd medical-cross-task-transfer

# Verify files
!ls -la
```

3. **Set up environment variables**:

```python
# In second cell - set your tokens
import os

# IMPORTANT: Use Kaggle Secrets (safer than hardcoding)
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ['HF_TOKEN'] = user_secrets.get_secret("HF_TOKEN")
os.environ['WANDB_API_KEY'] = user_secrets.get_secret("WANDB_API_KEY")

# Or if not using Kaggle Secrets (less secure):
# os.environ['HF_TOKEN'] = 'hf_your_token_here'
# os.environ['WANDB_API_KEY'] = 'your_wandb_key_here'
```

**To add Kaggle Secrets**:
- Go to https://www.kaggle.com/settings/secrets
- Add Secret: `HF_TOKEN` = your HuggingFace token
- Add Secret: `WANDB_API_KEY` = your Weights & Biases API key

### Method 2: Upload as Kaggle Dataset (For Datasets)

This is useful if you've already downloaded datasets locally:

1. **Zip your data**:

```bash
# On your local machine
cd h:\Projects\Research\Cross_Talk_Medical_LLM
tar -czf datasets.tar.gz data/raw/pubmedqa
```

2. **Upload to Kaggle**:
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload `datasets.tar.gz`
   - Title: "Medical MTL Datasets"
   - Click "Create"

3. **Use in notebook**:

```python
# In Kaggle notebook
!cp -r /kaggle/input/medical-mtl-datasets/data ./data
```

---

## Part 3: Install Dependencies in Kaggle

### Option 1: Install from requirements.txt (5-10 minutes)

```python
# In Kaggle notebook cell
!pip install -q -r requirements.txt
```

### Option 2: Install with uv (30 seconds - MUCH faster!)

```python
# Install uv first
!pip install -q uv

# Install dependencies with uv (blazing fast)
!uv pip install --system -r requirements.txt
```

**Note**: In Kaggle, use `--system` flag since we're not in a virtual environment.

---

## Part 4: Download Datasets in Kaggle

### PubMedQA (Working!)

```python
# This one works perfectly
!python scripts/download_datasets_hf.py --dataset pubmedqa
```

### BC5CDR, NCBI-Disease, DDI, GAD (Need Manual Download)

These datasets use deprecated HuggingFace loading scripts. **Solution**: Download from original sources.

#### Option A: Download in Kaggle (Recommended)

Create a new script `scripts/download_manual.py`:

```python
"""Manual dataset downloads for datasets not on HuggingFace Hub."""

import urllib.request
import zipfile
from pathlib import Path

def download_bc5cdr():
    """Download BC5CDR from BioCreative."""
    url = "https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip"
    output = Path("data/raw/bc5cdr_manual.zip")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading BC5CDR from {url}...")
    urllib.request.urlretrieve(url, output)

    print(f"Extracting...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(Path("data/raw/bc5cdr"))

    print(f"[OK] BC5CDR downloaded to data/raw/bc5cdr")

def download_ncbi_disease():
    """Download NCBI-Disease corpus."""
    url = "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBI_corpus.zip"
    output = Path("data/raw/ncbi_manual.zip")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading NCBI-Disease from {url}...")
    urllib.request.urlretrieve(url, output)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(Path("data/raw/ncbi_disease"))

    print(f"[OK] NCBI-Disease downloaded to data/raw/ncbi_disease")

# Usage
if __name__ == "__main__":
    download_bc5cdr()
    download_ncbi_disease()
    # Add others as needed
```

#### Option B: Use Kaggle Datasets

Search for pre-uploaded versions:
- https://www.kaggle.com/datasets → search "BC5CDR"
- https://www.kaggle.com/datasets → search "NCBI Disease"

Then add as input to your notebook.

---

## Part 5: Kaggle Notebook Structure

Here's the recommended structure for your Kaggle notebook:

### Cell 1: Setup

```python
# Clone repository
!git clone https://github.com/YOUR_USERNAME/medical-cross-task-transfer.git
%cd medical-cross-task-transfer

# Install uv for fast package installation
!pip install -q uv

# Install all dependencies
!uv pip install --system -r requirements.txt
```

### Cell 2: Environment Variables

```python
import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ['HF_TOKEN'] = user_secrets.get_secret("HF_TOKEN")
os.environ['WANDB_API_KEY'] = user_secrets.get_secret("WANDB_API_KEY")
os.environ['WANDB_ENTITY'] = 'your-username'
os.environ['WANDB_PROJECT'] = 'medical-mtl-kaggle'
```

### Cell 3: Download Datasets

```python
# Download PubMedQA (works via HuggingFace)
!python scripts/download_datasets_hf.py --dataset pubmedqa

# Manual downloads for others
!python scripts/download_manual.py  # You'll create this

# Or if you uploaded datasets as Kaggle dataset:
!cp -r /kaggle/input/medical-mtl-datasets/data ./data
```

### Cell 4: Verify Setup

```python
!python validate_setup.py
```

### Cell 5: Run First Experiment

```python
# Single-task baseline on PubMedQA
!python scripts/run_baseline.py --model bert-base-uncased --task pubmedqa

# Or hierarchical MTL (if all datasets ready)
!python scripts/run_experiment.py strategy=s3b_hierarchical task=all
```

---

## Part 6: Kaggle-Specific Optimizations

### 1. Session Timeout Protection

Kaggle free sessions timeout after **12 hours** or **9 hours of inactivity**.

**Solution**: Checkpoint frequently

```python
# Already implemented in your code!
# Checkpoints saved every 200 steps to prevent data loss
```

### 2. Disk Space Limits

Kaggle notebooks have **20GB** working directory limit.

**Solution**: Your code already handles this:
- Only saves LoRA adapters (10-50MB), not full models
- Keeps only last 2 checkpoints
- Uses 4-bit quantization

### 3. GPU Memory

T4 has **16GB VRAM**.

**Solution**: Your code includes:
- Auto batch size finder
- QLoRA 4-bit for 7-8B models
- VRAM monitoring

### 4. Internet Access

Kaggle notebooks have internet access **OFF by default**.

**Enable it**:
- Settings → Internet → **ON**

---

## Part 7: Update Code and Re-run

When you update code locally:

```bash
# On your local machine
git add .
git commit -m "Update: Added new feature"
git push
```

In Kaggle notebook:

```python
# Pull latest changes
!git pull origin main

# Re-run experiments
!python scripts/run_experiment.py strategy=s3b_hierarchical task=all
```

---

## Part 8: Save Results Back to GitHub

After experiments complete in Kaggle:

```python
# In Kaggle notebook - save results
!git config --global user.email "your@email.com"
!git config --global user.name "Your Name"

# Add results
!git add results/ figures/
!git commit -m "Results: S3b hierarchical MTL on PubMedQA"

# Push (requires GitHub token)
!git push https://YOUR_TOKEN@github.com/YOUR_USERNAME/medical-cross-task-transfer.git
```

**Better approach**: Download results from Kaggle and push from local machine.

---

## Part 9: Handling Large Datasets

If datasets are too large to push to GitHub:

### Option 1: Use Kaggle Datasets (Recommended)

1. Download datasets locally
2. Create Kaggle dataset: https://www.kaggle.com/datasets
3. Add as input to your notebook

### Option 2: Use Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "data/raw/*.zip"
git lfs track "data/raw/*.tar.gz"

# Add and push
git add .gitattributes
git add data/raw/
git commit -m "Add datasets with Git LFS"
git push
```

**Note**: GitHub free plan has 1GB LFS storage limit.

### Option 3: Download in Kaggle Every Time

```python
# Just download fresh in each Kaggle session
!python scripts/download_datasets_hf.py --all
!python scripts/download_manual.py
```

Since Kaggle has internet, this is often simplest.

---

## Part 10: Quick Start Commands

### Local Machine → GitHub

```bash
git add .
git commit -m "Your message"
git push
```

### Kaggle Notebook → First Cell

```python
!git clone https://github.com/YOUR_USERNAME/medical-cross-task-transfer.git
%cd medical-cross-task-transfer
!pip install -q uv
!uv pip install --system -r requirements.txt
```

### Kaggle Notebook → Run Experiment

```python
import os
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
os.environ['HF_TOKEN'] = user_secrets.get_secret("HF_TOKEN")
os.environ['WANDB_API_KEY'] = user_secrets.get_secret("WANDB_API_KEY")

# Download datasets
!python scripts/download_datasets_hf.py --dataset pubmedqa

# Run experiment
!python scripts/run_baseline.py --model bert-base-uncased --task pubmedqa
```

---

## Part 11: Troubleshooting

### "No space left on device"

```python
# Clear HuggingFace cache
!rm -rf ~/.cache/huggingface/*

# Clear pip cache
!pip cache purge
```

### "CUDA out of memory"

Your code has auto batch size finder:

```python
# It will automatically reduce batch size until it fits
# See: src/utils/auto_batch.py
```

### "Session timeout"

Enable checkpointing (already done!):

```python
# Your code saves checkpoints every 200 steps
# On session restart, resume from latest checkpoint
```

---

## Summary

**1. Push to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/medical-cross-task-transfer.git
git push -u origin main
```

**2. Pull in Kaggle**:
```python
!git clone https://github.com/YOUR_USERNAME/medical-cross-task-transfer.git
%cd medical-cross-task-transfer
```

**3. Install Dependencies**:
```python
!pip install -q uv
!uv pip install --system -r requirements.txt
```

**4. Set Secrets** (Kaggle Settings → Add HF_TOKEN, WANDB_API_KEY)

**5. Download Datasets**:
```python
!python scripts/download_datasets_hf.py --dataset pubmedqa
```

**6. Run Experiments**:
```python
!python scripts/run_baseline.py --model bert-base-uncased --task pubmedqa
```

---

**You're ready to run experiments on Kaggle's free T4 GPU!** 🚀

---

*Last updated: 2026-02-07*
