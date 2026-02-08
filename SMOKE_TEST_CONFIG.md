# Smoke Test Configuration
## Fast validation of all experiments before A100

## Phase 1: Kaggle Free - Smoke Tests (2-3 hours)

### Purpose
- Test all 7 models × 8 datasets = 56 combinations
- Catch errors before paying for A100
- Verify: data loading, model loading, training loop, metrics, CSV export
- Total cost: $0
- Total time: 2-3 hours

### Configuration

```python
# Ultra-fast smoke test
SMOKE_TEST_CONFIG = {
    "experiment_type": "single_task",
    "max_samples_per_dataset": 50,  # Tiny! Just to test
    "num_epochs": 1,
    "batch_size": 16,  # Conservative for safety
    "learning_rate": 2e-5,
    "max_length": 128,  # Shorter for speed
    "warmup_steps": 10,
    "weight_decay": 0.01,

    # Minimal checkpointing
    "save_strategy": "steps",
    "save_steps": 50,
    "keep_last_n_checkpoints": 1,
    "resume_from_checkpoint": False,  # Fresh start

    # Minimal evaluation
    "eval_strategy": "steps",
    "eval_steps": 25,

    # Early stopping disabled for smoke test
    "use_early_stopping": False,

    # Token tracking enabled
    "track_tokens": True,

    # Logging
    "use_wandb": False,
    "logging_steps": 10,
}
```

### Expected Time per Experiment

| Component | Time |
|-----------|------|
| Model download | 30s (first time only) |
| Data loading | 5s |
| Training (50 samples, 1 epoch) | 1-2 min |
| Evaluation | 10s |
| Save checkpoint | 5s |
| Save CSV | 1s |
| **Total** | **2-3 minutes** |

### Total Smoke Test Time

```
56 experiments × 2.5 min = 140 min = 2.3 hours
```

✅ **Fits comfortably in Kaggle's 30h/week limit!**

## Test Matrix

### All 7 Models × 8 Datasets = 56 Tests

| Model | BC2GM | JNLPBA | ChemProt | DDI | GAD | HoC | PubMedQA | BIOSSES |
|-------|-------|--------|----------|-----|-----|-----|----------|---------|
| BERT-base | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| RoBERTa-base | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BioBERT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PubMedBERT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Clinical-BERT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BlueBERT | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| BioMed-RoBERTa | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## What Gets Validated

### 1. Model Loading ✓
- All 7 models download correctly
- No incompatibility issues
- Correct architecture loaded

### 2. Data Loading ✓
- All 8 datasets load from pickle
- Tokenization works correctly
- Label maps work for all datasets
- No missing keys or corrupted data

### 3. Training Loop ✓
- Forward pass works
- Backward pass works
- Optimizer step works
- No CUDA OOM errors
- Gradient computation correct

### 4. Metrics Computation ✓
- NER metrics (seqeval) work correctly
- Classification metrics (sklearn) work correctly
- No label mismatch errors
- F1/Precision/Recall calculated

### 5. Checkpointing ✓
- Checkpoints save successfully
- Checkpoint directory created
- Model state saved correctly

### 6. CSV Export ✓
- Results append to all_experiments.csv
- All 24 columns populated correctly
- No CSV formatting errors
- File not corrupted

### 7. Early Stopping (tested separately) ✓
- Patience mechanism works
- Best model restored
- Training stops correctly

## Expected Output

After 2-3 hours, you'll have:

### 1. Complete CSV File
```csv
experiment_id,model_name,dataset,test_f1,test_precision,test_recall,...
20260208_120001,bert-base-uncased,bc2gm,0.45,0.42,0.48,...
20260208_120230,bert-base-uncased,jnlpba,0.38,0.35,0.41,...
...
(56 rows total)
```

### 2. Validation Report
```
✅ 56/56 experiments completed successfully
✅ All models loaded without errors
✅ All datasets processed correctly
✅ All metrics computed successfully
✅ CSV export working
✅ Checkpointing working

Ready for A100 full runs!
```

### 3. Error Report (if any)
```
❌ Model X failed on Dataset Y:
   Error: RuntimeError: CUDA out of memory
   Solution: Reduce batch size to 8

❌ Dataset Z has label mismatch:
   Error: KeyError: 'ner_tags'
   Solution: Update LABEL_MAPS dictionary
```

## Troubleshooting Common Issues

### Issue 1: Model Download Timeout
```python
# Add retries
from huggingface_hub import hf_hub_download
import time

for attempt in range(3):
    try:
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        break
    except Exception as e:
        if attempt < 2:
            print(f"Retry {attempt+1}/3...")
            time.sleep(5)
        else:
            raise e
```

### Issue 2: CUDA OOM on Specific Model
```python
# Reduce batch size for specific models
if "roberta" in model_name.lower():
    CONFIG['batch_size'] = 8  # RoBERTa uses more memory
```

### Issue 3: Label Mismatch
```python
# Already fixed in notebook with LABEL_MAPS
# If still occurs, check dataset pickle file
```

## After Smoke Tests Pass

### Checklist Before A100
- [ ] All 56 experiments completed without errors
- [ ] CSV file has 56 rows with valid data
- [ ] No CUDA OOM errors
- [ ] All metrics calculated correctly
- [ ] No label mismatch warnings
- [ ] Checkpoints saved successfully
- [ ] Results look reasonable (F1 > 0.3 even for 50 samples)

### If Any Test Fails
1. Fix the issue in notebook
2. Re-run ONLY the failed experiments
3. Verify fix works
4. Commit fix to GitHub
5. When all pass → proceed to A100

### If All Pass ✅
You're ready for Phase 2: A100 Full Runs!

---

## Phase 2: A100 Commercial - Full Runs (2 days)

### Configuration for A100

```python
# Full production configuration
A100_FULL_CONFIG = {
    "experiment_type": "single_task",
    "max_samples_per_dataset": None,  # Use ALL data
    "num_epochs": 10,  # Max (early stopping will stop earlier)
    "batch_size": 64,  # A100 can handle this
    "learning_rate": 2e-5,
    "max_length": 512,
    "warmup_steps": 500,
    "weight_decay": 0.01,

    # Aggressive checkpointing for interruptible
    "save_strategy": "steps",
    "save_steps": 100,  # Every ~2 min
    "keep_last_n_checkpoints": 2,
    "resume_from_checkpoint": True,  # Auto-resume

    # Evaluation
    "eval_strategy": "steps",
    "eval_steps": 250,

    # Early stopping enabled
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.0001,

    # Token tracking CRITICAL
    "track_tokens": True,

    # Logging
    "use_wandb": False,
    "logging_steps": 50,
}
```

### Expected Performance on A100

| Experiment | Time | Cost @ $0.143/hr |
|------------|------|------------------|
| BC2GM | 0.6h | $0.086 |
| JNLPBA | 0.7h | $0.100 |
| ChemProt | 1.0h | $0.143 |
| DDI | 0.9h | $0.129 |
| GAD | 0.3h | $0.043 |
| HoC | 0.2h | $0.029 |
| PubMedQA | 2.0h | $0.286 |
| BIOSSES | 0.1h | $0.014 |
| **Average** | **0.7h** | **$0.10** |

### Total Cost & Time

```
56 experiments × 0.7h = 40 GPU hours
40h × $0.143/hr = $5.72

Calendar time: 1.7 days (continuous)
```

## Summary

### Phase 1: Kaggle Free (Smoke Tests)
- Time: 2-3 hours
- Cost: $0
- Purpose: Catch all errors
- Samples: 50 per dataset
- Outcome: Validated setup

### Phase 2: A100 Commercial (Full Runs)
- Time: 40 GPU hours (2 days)
- Cost: $5.72
- Purpose: Publication-quality results
- Samples: All data
- Outcome: Complete results for paper

### Total Investment
- Time: 2 days
- Cost: $5.72
- Risk: Minimal (everything tested first)
- Confidence: High (all bugs caught)

---

**This is the optimal strategy: Test everything thoroughly for free, then pay only for proven, working experiments!**
