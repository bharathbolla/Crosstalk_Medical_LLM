# 🔥 Multi-Model Smoke Test Guide
## Test All 7 Models Before Full Experiments

**Purpose**: Validate all 7 BERT models work correctly before running full experiments
**Time**: ~15 minutes (2 min per model × 7 models)
**Expected**: All models F1 > 0.30 on BC2GM

---

## 🎯 Why Test All Models?

**Problem**: You don't want to discover a model is broken after 3 hours of training!

**Solution**: Run 15-minute smoke test on all models first

**Benefits**:
- ✅ Catch model-specific issues early (RoBERTa, etc.)
- ✅ Verify all 7 models download correctly
- ✅ Confirm tokenization works for each
- ✅ Detect GPU/memory issues per model
- ✅ Get baseline F1 scores for comparison

---

## 📋 Two Approaches

### **Approach 1: In Kaggle Notebook** ⭐ Recommended

**Use when**: Testing on Kaggle before full experiments

**Time**: ~15 minutes on Kaggle T4

**How**:
1. Open KAGGLE_COMPLETE.ipynb
2. Replace Cell 4 with code from `CELL_4_ALL_MODELS_SMOKE_TEST.py`
3. Add new cell after Cell 9 with code from `CELL_MULTI_MODEL_TRAINER.py`
4. Set `TEST_ALL_MODELS = True`
5. Run all cells

**Result**: CSV file with results for all 7 models

---

### **Approach 2: Standalone Script** ⭐ Run locally first

**Use when**: Testing locally before uploading to Kaggle

**Time**: ~15 minutes (or ~30 min without GPU)

**How**:
```bash
python validate_all_models_smoke_test.py
```

**Output**:
- Progress for each model
- Summary table
- CSV file: `results/model_validation_YYYYMMDD_HHMMSS.csv`

---

## 📊 Expected Results

| Model | Expected F1 | Time | Notes |
|-------|-------------|------|-------|
| BioBERT | 0.35-0.45 | 2 min | Good baseline |
| BlueBERT | 0.35-0.50 | 2 min | Often best |
| PubMedBERT | 0.35-0.45 | 2 min | Competitive |
| BioMed-RoBERTa | 0.30-0.40 | 2 min | Needs add_prefix_space |
| Clinical-BERT | 0.30-0.40 | 2 min | Clinical focus |
| RoBERTa | 0.25-0.35 | 2 min | General baseline |
| BERT | 0.20-0.30 | 2 min | Weakest (no medical) |

**Pass criteria**: F1 > 0.30 for all models

**Note**: Smoke test scores are LOWER than full training (1 epoch vs 10)

---

## 🚀 Quick Start (Approach 1: Kaggle)

### **Step 1: Modify Cell 4**

Replace Cell 4 content with:

```python
# ============================================
# 🔥 MULTI-MODEL SMOKE TEST
# ============================================

import sys
import io
if sys.platform != 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('\n' + '='*60)
print('🔥 MULTI-MODEL SMOKE TEST')
print('='*60)

# ⭐ Set to True to test all models
TEST_ALL_MODELS = True  # Change to False for single model

if TEST_ALL_MODELS:
    # All 7 models
    all_models = [
        'dmis-lab/biobert-v1.1',
        'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'allenai/biomed_roberta_base',
        'emilyalsentzer/Bio_ClinicalBERT',
        'roberta-base',
        'bert-base-uncased',
    ]

    CONFIG['all_models_to_test'] = all_models
    CONFIG['test_results'] = []

    # Smoke test config
    CONFIG['max_samples_per_dataset'] = 50
    CONFIG['num_epochs'] = 1
    CONFIG['batch_size'] = 16
    CONFIG['max_length'] = 128
    CONFIG['datasets'] = ['bc2gm']

    print(f'\n✅ Will test {len(all_models)} models')
    print('⏱️  Expected time: ~15 minutes')
else:
    # Single model (original)
    CONFIG['max_samples_per_dataset'] = 50
    CONFIG['num_epochs'] = 1
    CONFIG['batch_size'] = 16

print('='*60)
```

### **Step 2: Insert Multi-Model Trainer Cell**

**Insert as Cell 10** (before current Cell 10 "Train Model"):

Copy code from `CELL_MULTI_MODEL_TRAINER.py`

This cell:
- Loops through all 7 models
- Trains each for 1 epoch
- Collects results
- Saves CSV
- Shows summary

### **Step 3: Run**

1. Run all cells
2. Wait ~15 minutes
3. Check results table

### **Step 4: Interpret Results**

```
📊 MULTI-MODEL SMOKE TEST SUMMARY
============================================================
Results:
       model_short     f1  precision  recall  time_seconds
        biobert-v1.1  0.42       0.45    0.39           120
      bluebert_pub…  0.48       0.51    0.45           125
   BiomedNLP-PubM…  0.44       0.47    0.41           122
   biomed_roberta…  0.38       0.40    0.36           118
   Bio_ClinicalBE…  0.36       0.38    0.34           120
         roberta-b…  0.32       0.35    0.30           115
     bert-base-unc…  0.28       0.30    0.26           110

Passed: 6/7 ✅
Failed: 1/7 ❌
============================================================
```

**Analysis**:
- ✅ 6 models passed (F1 > 0.30)
- ❌ BERT failed (F1 = 0.28 < 0.30) - **Expected!** (no medical training)
- ✅ BlueBERT best (F1 = 0.48)
- ✅ All medical models work

**Decision**: Proceed with full experiments! ✅

---

## 🚀 Quick Start (Approach 2: Standalone)

### **Step 1: Run Script**

```bash
cd h:\Projects\Research\Cross_Talk_Medical_LLM
python validate_all_models_smoke_test.py
```

### **Step 2: Monitor Progress**

```
============================================================
COMPREHENSIVE MODEL VALIDATION
============================================================

Testing all 7 BERT models on BC2GM
Time: ~15 minutes
============================================================

############################################################
MODEL 1/7
############################################################

============================================================
Testing: dmis-lab/biobert-v1.1
============================================================
Train samples: 50
Val samples: 10

🤖 Loading model for bc2gm...
   Model: dmis-lab/biobert-v1.1
   Task type: ner
   Number of labels: 3
   ✅ Model loaded

🚀 Training...
Epoch 1/1: 100%|██████████| ...
📊 Evaluating...

✅ Complete: F1 = 0.4200 (PASS)

[... continues for all 7 models ...]
```

### **Step 3: Check Summary**

```
============================================================
📊 VALIDATION SUMMARY
============================================================

Results:
       model_short     f1  precision  recall  time_seconds status
        biobert-v1.1  0.42       0.45    0.39           120   PASS
...

Passed: 6/7 ✅
Failed: 1/7 ❌
Errors: 0/7 💥
============================================================

💾 Results saved: results/model_validation_20260208_143052.csv

🎉 ALL MODELS PASSED!
✅ Ready for full experiments
```

---

## 📊 Results Interpretation

### **All Passed (6-7/7)**
✅ **Action**: Proceed with full experiments
- All models working correctly
- Tokenization verified
- GPU memory sufficient
- Ready for Phase 2

### **Some Failed (3-5/7)**
⚠️ **Action**: Investigate failures
- Check error messages
- Verify model downloads
- Test failed models individually
- Check GPU memory

### **Most Failed (0-2/7)**
❌ **Action**: Fix environment first
- Check data loading
- Verify COMPLETE_FIXED_*.py loaded
- Check GPU availability
- Run `test_pickle_load.py`
- Check TROUBLESHOOTING_GUIDE.md

---

## 🔧 Troubleshooting

### **Issue: RoBERTa models fail**
```
Error: AssertionError: add_prefix_space=True required
```

**Fix**: Already handled in code!
```python
if 'roberta' in model_name.lower():
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
```

### **Issue: BERT fails (F1 = 0.28)**
**This is expected!** BERT-base has no medical training.
- Expected F1: 0.20-0.30 (just barely passing)
- Not a problem, just shows medical models are better

### **Issue: Out of memory**
```
RuntimeError: CUDA out of memory
```

**Fix**: Reduce batch size in code
```python
CONFIG['batch_size'] = 8  # Instead of 16
```

### **Issue: Model download fails**
```
ConnectionError: Failed to download
```

**Fix**: Check internet connection, try again
Or skip that model (comment out from list)

---

## 💾 Output Files

### **CSV File**
Location: `results/model_validation_YYYYMMDD_HHMMSS.csv`

Columns:
- `model`: Full model name
- `model_short`: Short name
- `dataset`: Dataset tested (bc2gm)
- `f1`, `precision`, `recall`: Metrics
- `train_loss`, `eval_loss`: Losses
- `time_seconds`: Training time
- `status`: PASS/FAIL/ERROR

**Use for**:
- Compare models
- Track over time
- Share results
- Publication tables

---

## 📈 Next Steps

### **After All Pass**:
1. ✅ Proceed to full training
2. Set `SMOKE_TEST = False`
3. Set `num_epochs = 10`
4. Run Phase 2 experiments

### **After Some Fail**:
1. Check which models failed
2. Investigate error messages
3. Fix issues
4. Re-run smoke test
5. When all pass → proceed

---

## 🎯 Best Practices

### **Before Full Experiments**:
- ✅ Run multi-model smoke test
- ✅ Verify all 7 models pass
- ✅ Save results CSV
- ✅ Check BERT is weakest (confirms medical models work)

### **During Development**:
- ✅ Run after code changes
- ✅ Verify fixes don't break other models
- ✅ Test new tokenization approaches

### **Before Sharing**:
- ✅ Run to generate baseline results
- ✅ Include CSV in supplementary materials
- ✅ Show all models were validated

---

## 🎉 Success Criteria

**Ready for Phase 2 when**:
- ✅ 6-7 out of 7 models pass (F1 > 0.30)
- ✅ All medical models pass (BioBERT, BlueBERT, etc.)
- ✅ RoBERTa models work (tokenization correct)
- ✅ No GPU memory errors
- ✅ Results saved to CSV
- ✅ Total time < 20 minutes

**Then you can confidently start 56 full experiments! (7 models × 8 tasks)**

---

## 📞 Quick Reference

### **Approach 1 (Kaggle)**:
```python
# Cell 4: Set TEST_ALL_MODELS = True
# Cell 10: Insert multi-model trainer loop
# Run all cells → Wait 15 min → Check results
```

### **Approach 2 (Local)**:
```bash
python validate_all_models_smoke_test.py
# Wait 15 min → Check CSV → Proceed
```

### **Expected Time**:
- Per model: 2 minutes
- All 7 models: 15 minutes
- With errors/retries: 20 minutes

### **Expected F1** (smoke test):
- Medical models: 0.35-0.50
- General models: 0.25-0.35
- Pass threshold: > 0.30

---

**Remember**: Smoke test F1 is LOWER than full training!
- Smoke: 1 epoch, 50 samples → F1 = 0.30-0.50
- Full: 10 epochs, all samples → F1 = 0.80-0.85

**The goal is just to verify everything works, not to get SOTA!** ✅
