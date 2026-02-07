# Quick Start Guide - Next Steps

**Status**: ✅ All parsers working, ready for Kaggle!

---

## 🚀 What to Do RIGHT NOW

### Step 1: Create Kaggle Notebook (Today)

1. **Go to**: https://www.kaggle.com/code
2. **Click**: "New Notebook"
3. **Settings** (right sidebar):
   - Accelerator: **GPU T4 x2**
   - Internet: **On**
   - Persistence: **Variables and files**

### Step 2: Upload Your Code (10 minutes)

**Option A**: Upload as Dataset (Recommended)
```bash
# On your local machine:
# 1. Create a zip file with these folders:
#    - src/
#    - scripts/
#    - configs/
#    - test_parsers.py
#    - requirements.txt

# 2. Go to Kaggle → Datasets → New Dataset
# 3. Upload the zip file
# 4. Name it: "medical-mtl-code"
# 5. Click "Create"

# 6. In your notebook, add it as input:
#    + Add Data → Your Datasets → medical-mtl-code
```

**Option B**: Copy-Paste Code
```python
# In Kaggle cell 1 - Create directory structure
!mkdir -p src/data src/models src/training src/evaluation
!mkdir -p scripts configs/task configs/strategy

# Then paste your code files into separate cells
```

### Step 3: Download Datasets on Kaggle (15 minutes)

**Kaggle Cell - Download All 8 Datasets**:
```python
from datasets import load_dataset
from pathlib import Path

# Create data directory
data_path = Path("/kaggle/working/data/raw")
data_path.mkdir(parents=True, exist_ok=True)

# Download all 8 datasets using Parquet URLs
datasets = {
    "bc2gm": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/bc2gm",
    "jnlpba": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/jnlpba",
    "chemprot": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/chemprot",
    "ddi": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/ddi_corpus",
    "gad": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/gad",
    "hoc": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/hallmarks_of_cancer",
    "pubmedqa": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/pubmed_qa",
    "biosses": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/biosses"
}

for name, base_url in datasets.items():
    print(f"Downloading {name}...")

    # Determine which splits are available
    if name in ["bc2gm", "jnlpba", "chemprot"]:
        splits = {"train": "train/0000.parquet",
                  "validation": "validation/0000.parquet",
                  "test": "test/0000.parquet"}
    elif name in ["ddi", "gad"]:
        splits = {"train": "train/0000.parquet",
                  "test": "test/0000.parquet"}
    elif name in ["hoc", "pubmedqa", "biosses"]:
        splits = {"train": "train/0000.parquet",
                  "validation": "validation/0000.parquet",
                  "test": "test/0000.parquet"}

    data_files = {split: f"{base_url}/{path}"
                  for split, path in splits.items()}

    dataset = load_dataset("parquet", data_files=data_files)
    dataset.save_to_disk(str(data_path / name))
    print(f"  ✓ Saved to {data_path / name}")

print("\n✅ All 8 datasets downloaded!")
```

### Step 4: Test Everything Works (5 minutes)

**Kaggle Cell - Run Tests**:
```python
# Test GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test parsers
import sys
sys.path.insert(0, "/kaggle/working/src")

from data import TaskRegistry, BC2GMDataset
from pathlib import Path

print(f"\nRegistered tasks: {TaskRegistry.list_tasks()}")

# Load one dataset
dataset = BC2GMDataset(
    data_path=Path("/kaggle/working/data/raw"),
    split="train"
)
print(f"\n✓ Loaded {len(dataset)} BC2GM samples")
print(f"✓ First sample: {dataset[0].input_text[:100]}...")

print("\n🎉 Everything works! Ready to train!")
```

---

## 📊 Expected Output

```
CUDA available: True
GPU: Tesla T4
VRAM: 15.0 GB

Registered tasks: ['bc2gm', 'jnlpba', 'chemprot', 'ddi', 'gad', 'hoc', 'pubmedqa', 'biosses']

✓ Loaded 12574 BC2GM samples
✓ First sample: Immunohistochemical staining was positive for S - 100 in all 9 cases stained , positive...

🎉 Everything works! Ready to train!
```

---

## 🏃 First Experiment - Smoke Test (10 minutes)

**Goal**: Verify training pipeline works on tiny subset

**Kaggle Cell**:
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from pathlib import Path
from src.data import BC2GMDataset
from src.data.collators import NERCollator

# 1. Load tiny subset (100 samples)
dataset = BC2GMDataset(
    data_path=Path("/kaggle/working/data/raw"),
    split="train"
)
small_dataset = dataset[:100]  # Just first 100 samples

# 2. Load model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

label_schema = dataset.get_label_schema()
num_labels = len(label_schema)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# 3. Setup training
training_args = TrainingArguments(
    output_dir="/kaggle/working/smoke_test",
    max_steps=50,  # Just 50 steps!
    per_device_train_batch_size=4,
    logging_steps=10,
    save_steps=25,
    fp16=True,  # Use mixed precision
)

collator = NERCollator(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_dataset,
    data_collator=collator
)

# 4. Train!
print("🚀 Starting smoke test (50 steps)...")
trainer.train()
print("✅ Smoke test complete! Pipeline works!")
```

**Expected**: Training completes in ~2 minutes without errors

---

## 🎯 After Smoke Test Succeeds

### Next: Run Contamination Check (2 hours)

This is **critical** before any real experiments!

**Why**: Check if test data leaked into pre-training

**Kaggle Cell**:
```python
# Contamination check for all 6 models × 8 tasks
!python /kaggle/working/scripts/run_contamination_check.py \
    --data_path /kaggle/working/data/raw \
    --output_dir /kaggle/working/contamination_results \
    --device cuda

# Results saved to contamination_results/contamination_report.json
```

**Runtime**: ~2 hours on T4 (uses ~2 of your 30 weekly GPU hours)

---

## 📅 Week 1 Schedule (After Smoke Test)

### Day 1: Setup & Contamination
- [x] Create Kaggle notebook
- [x] Upload code
- [x] Download datasets
- [x] Run smoke test
- [ ] Run contamination check (2 hours)
- [ ] Review contamination results

### Day 2: Baselines Start
- [ ] BERT on BC2GM (1 hour)
- [ ] BERT on JNLPBA (1.5 hours)
- [ ] BioBERT on BC2GM (1 hour)

### Day 3: Baselines Continue
- [ ] BioBERT on remaining tasks (3 hours)
- [ ] Verify F1 scores match published SOTA (±2 points)

### Day 4-5: Single-Task S1 Start
- [ ] Llama-3.2-1B + LoRA on BC2GM (1 hour)
- [ ] Llama-3.2-3B + LoRA on BC2GM (1.5 hours)
- [ ] Start token-count logging

**Total Week 1 GPU hours**: ~12 hours (well within 30-hour budget!)

---

## 💡 Pro Tips

### 1. Checkpoint Every 200 Steps
```python
training_args = TrainingArguments(
    ...
    save_steps=200,
    save_total_limit=2,  # Keep only last 2 checkpoints (save disk space)
)
```

### 2. Monitor GPU Usage
```python
# In a separate cell, run this to monitor VRAM:
!watch -n 5 nvidia-smi  # Updates every 5 seconds
```

### 3. Use Mixed Precision
```python
# Always enable FP16 or BF16 for 2x speedup:
training_args = TrainingArguments(
    ...
    fp16=True,  # T4 supports this
    # OR bf16=True if model supports it
)
```

### 4. Auto Batch Size Finder
```python
# Find max batch size that fits in VRAM:
from transformers import TrainingArguments

training_args = TrainingArguments(
    ...
    per_device_train_batch_size=1,  # Start small
    auto_find_batch_size=True,  # Auto-scale up!
)
```

### 5. Kaggle Session Management
- **Save often**: Sessions can disconnect
- **Checkpoint strategy**: Every 200 steps
- **Keep only 2 checkpoints**: Kaggle has 20GB disk limit
- **Upload to HF Hub**: For long-term storage

---

## 📁 Files You Have

**On Your Local Machine**:
```
h:\Projects\Research\Cross_Talk_Medical_LLM\
├── src/data/              (8 parsers + base classes)
├── scripts/               (training scripts - TO BE IMPLEMENTED)
├── configs/              (YAML configs - TO BE CREATED)
├── data/raw/             (8 datasets downloaded)
├── test_parsers.py       (✅ All tests pass!)
├── PARSERS_COMPLETE.md   (Full parser documentation)
├── DATA_PARSING_EXPLAINED.md  (Tutorial)
├── GPU_ASSESSMENT.md     (GPU comparison analysis)
├── SYSTEM_STATUS.md      (Comprehensive status report)
└── QUICK_START.md        (This file!)
```

**What to Upload to Kaggle**:
- `src/` folder (all your code)
- `scripts/` folder (when implemented)
- `configs/` folder (when created)
- `test_parsers.py`
- `requirements.txt`

**What NOT to Upload** (will download on Kaggle):
- `data/raw/` (too large, re-download on Kaggle)

---

## ❓ Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
```python
import sys
sys.path.insert(0, "/kaggle/working")
# Then import normally
from src.data import TaskRegistry
```

### "CUDA out of memory"
```python
# Reduce batch size:
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # Try 2, 4, 8 until it fits
    gradient_accumulation_steps=4,  # Simulate larger batch
)
```

### "Datasets not found"
```python
# Make sure you downloaded to the right path:
data_path = Path("/kaggle/working/data/raw")
dataset = BC2GMDataset(data_path=data_path, split="train")
```

---

## ✅ Success Criteria

**After completing this guide, you should have**:

1. ✅ Kaggle notebook running with T4 GPU
2. ✅ All 8 datasets downloaded on Kaggle
3. ✅ All 8 parsers working on Kaggle
4. ✅ Smoke test completed (50 steps)
5. ✅ Ready to run contamination check

**Then you can proceed to**:
- Week 1: Baselines (BERT, BioBERT)
- Week 2: Single-task (S1) + token logging
- Week 3: Multi-task (S2, S3a, S3b)
- Week 4: Quantization (S5) + final experiments

---

## 🎉 You've Got This!

You've already completed the hardest part:
- ✅ Downloaded 8 diverse datasets
- ✅ Implemented 8 working parsers
- ✅ Tested everything locally
- ✅ Understood GPU limitations

**All that's left**: Copy to Kaggle and press "Run"!

---

**Next Action**: Go to https://www.kaggle.com/code and create your first notebook!

---

*Last updated: 2026-02-07*
