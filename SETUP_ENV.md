# Environment Setup Guide

This guide explains how to set up your environment variables for the Medical Cross-Task Transfer project.

---

## Quick Setup

1. **Copy the template**:
   ```bash
   cp .env.template .env
   ```

2. **Edit `.env` with your actual API keys** (see below for how to get them)

3. **Verify setup**:
   ```bash
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('HF_TOKEN:', 'SET' if os.getenv('HF_TOKEN') else 'MISSING')"
   ```

---

## Required Environment Variables

### 1. HuggingFace Token (REQUIRED)

**Why**: Download gated models like Llama, Gemma, Mistral

**Get from**: https://huggingface.co/settings/tokens

**Steps**:
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Name it "medical-mtl-research"
4. Select "Read" permission
5. Copy the token (starts with `hf_...`)
6. Add to `.env`:
   ```bash
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

**Accept model licenses** (one-time per model):
- Llama: https://huggingface.co/meta-llama/Llama-3.2-3B
- Gemma: https://huggingface.co/google/gemma-2-2b
- Phi-3: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
- Mistral: https://huggingface.co/mistralai/Mistral-7B-v0.3
- BioMistral: https://huggingface.co/BioMistral/BioMistral-7B

Click "Accept" on each model page, then your token will work.

---

### 2. Weights & Biases Token (REQUIRED for logging)

**Why**: Track experiments, log metrics, visualize results

**Get from**: https://wandb.ai/authorize

**Steps**:
1. Create account at https://wandb.ai/ (free for academic use)
2. Go to https://wandb.ai/authorize
3. Copy your API key
4. Add to `.env`:
   ```bash
   WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   WANDB_ENTITY=your-username
   WANDB_PROJECT=medical-mtl
   ```

**Disable wandb** (if you prefer not to use it):
```bash
# In .env
WANDB_MODE=disabled

# Or in code
import os
os.environ["WANDB_MODE"] = "disabled"
```

---

## Optional Environment Variables

### 3. OpenAI API Key (OPTIONAL - only for GPT-4 baseline)

**Why**: Run GPT-4o-mini reference baseline (not required for main experiments)

**Get from**: https://platform.openai.com/api-keys

**Steps**:
1. Create account at https://platform.openai.com/
2. Go to API keys section
3. Create new secret key
4. Add to `.env`:
   ```bash
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

**Cost**: ~$0.15 per 1M input tokens, ~$0.60 per 1M output tokens for GPT-4o-mini

**Note**: Only needed for `scripts/run_gpt4_baseline.py`

---

### 4. Kaggle API (OPTIONAL)

**Why**: Download datasets from Kaggle

**Get from**: https://www.kaggle.com/settings

**Steps**:
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New Token"
4. Download `kaggle.json`
5. Extract credentials and add to `.env`:
   ```bash
   KAGGLE_USERNAME=your-kaggle-username
   KAGGLE_KEY=xxxxxxxxxxxxxxxx
   ```

---

### 5. PhysioNet Credentials (OPTIONAL - for dataset access)

**Why**: Access SemEval medical NLP datasets

**Get from**: https://physionet.org/

**Steps**:
1. Create account at https://physionet.org/register/
2. Complete CITI training (required for credentialed access)
3. Request access to specific datasets
4. Add to `.env`:
   ```bash
   PHYSIONET_USERNAME=your-username
   PHYSIONET_PASSWORD=your-password
   ```

**Note**: PhysioNet access requires 1-2 week approval. Apply early!

---

## Cache Directories (OPTIONAL)

Control where models and datasets are cached:

```bash
# HuggingFace cache (models)
HF_HOME=./cache/huggingface
TRANSFORMERS_CACHE=./cache/transformers

# Data directories
DATA_DIR=./data
RESULTS_DIR=./results
CHECKPOINTS_DIR=./checkpoints
```

**Default locations** (if not set):
- Models: `~/.cache/huggingface/`
- Datasets: `~/.cache/huggingface/datasets/`

**Tip**: Set custom cache if your home directory has limited space.

---

## Compute Settings (OPTIONAL)

```bash
# GPU selection
CUDA_VISIBLE_DEVICES=0  # Use first GPU
# CUDA_VISIBLE_DEVICES=0,1  # Use multiple GPUs

# CPU threads for data loading
OMP_NUM_THREADS=8

# PyTorch CUDA allocator
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Example `.env` File

Here's a complete example with all variables:

```bash
# Required
HF_TOKEN=hf_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890
WANDB_API_KEY=1234567890abcdef1234567890abcdef12345678
WANDB_ENTITY=myusername
WANDB_PROJECT=medical-mtl

# Optional
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
KAGGLE_USERNAME=myusername
KAGGLE_KEY=abcdef1234567890

# Paths
HF_HOME=./cache/huggingface
DATA_DIR=./data
RESULTS_DIR=./results
CHECKPOINTS_DIR=./checkpoints

# Compute
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=8
```

---

## Verification

### Check if environment variables are loaded:

```bash
python -c "
from dotenv import load_dotenv
import os

load_dotenv()

print('HF_TOKEN:', 'SET ✓' if os.getenv('HF_TOKEN') else 'MISSING ✗')
print('WANDB_API_KEY:', 'SET ✓' if os.getenv('WANDB_API_KEY') else 'MISSING ✗')
print('OPENAI_API_KEY:', 'SET ✓' if os.getenv('OPENAI_API_KEY') else 'MISSING ✗')
"
```

### Test HuggingFace authentication:

```bash
python -c "
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
token = os.getenv('HF_TOKEN')

try:
    login(token=token)
    print('✓ HuggingFace authentication successful!')
except Exception as e:
    print(f'✗ Authentication failed: {e}')
"
```

### Test wandb authentication:

```bash
python -c "
import wandb
from dotenv import load_dotenv
import os

load_dotenv()

try:
    wandb.login(key=os.getenv('WANDB_API_KEY'))
    print('✓ Weights & Biases authentication successful!')
except Exception as e:
    print(f'✗ Authentication failed: {e}')
"
```

---

## Security Best Practices

### ⚠️ NEVER commit `.env` to git!

The `.gitignore` file already excludes `.env`, but double-check:

```bash
# Verify .env is ignored
git check-ignore .env
# Should output: .env

# If not, add it:
echo ".env" >> .gitignore
```

### Store tokens securely

- Don't share your `.env` file
- Don't paste tokens in public forums/Slack/Discord
- Regenerate tokens if accidentally exposed
- Use read-only tokens when possible

### Alternative: Use system environment variables

Instead of `.env`, you can set variables in your shell:

```bash
# Linux/Mac
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Windows PowerShell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:WANDB_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Windows CMD
set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
set WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Troubleshooting

### "HF_TOKEN not found" error

1. Check `.env` exists: `ls -la .env`
2. Check `.env` has correct format (no quotes around values)
3. Restart Python session after editing `.env`
4. Verify with: `python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('HF_TOKEN'))"`

### "Access denied" for Llama/Gemma

1. Accept model license on HuggingFace (links above)
2. Wait 5-10 minutes for access to propagate
3. Try again

### "wandb login failed"

1. Check API key is correct
2. Try logging in via CLI: `wandb login`
3. If all else fails, disable wandb: `WANDB_MODE=disabled`

### "Module 'dotenv' not found"

```bash
pip install python-dotenv
```

---

## Next Steps

After setting up `.env`:

1. ✅ **Install dependencies**: `pip install -e .`
2. ✅ **Verify installation**: `python verify_imports.py`
3. ✅ **Test model loading**: See NEXT_STEPS.md
4. ✅ **Run first experiment**: See CONFIGS_AND_SCRIPTS_COMPLETE.md

---

*Last updated: 2026-02-07*
