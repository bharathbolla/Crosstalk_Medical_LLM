# ⚡ Quick Start Checklist - Next Experiment

**Before starting ANY experiment, run this 5-minute checklist!**

---

## ☑️ Pre-Flight (5 minutes)

### 1. Verify Data (1 minute)
```bash
python test_pickle_load.py
python tests/test_data_loading.py
```
**Expected**: ✅ All 8 datasets loaded

### 2. Verify Notebook (10 seconds)
```bash
python validate_notebook.py KAGGLE_COMPLETE.ipynb
```
**Expected**: ✅ Validation passed

### 3. Upload to Kaggle
- Upload [KAGGLE_COMPLETE.ipynb](KAGGLE_COMPLETE.ipynb)
- Enable GPU (T4)

### 4. Smoke Test (2 minutes)
```python
# Cell 4: Keep this
SMOKE_TEST = True

# Cell 3: Start with BC2GM
CONFIG['datasets'] = ['bc2gm']
```
**Run All Cells**

**Expected**: F1 > 0.30 in ~2 minutes

❌ **If F1 < 0.30**: STOP! Check [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)

---

## ✅ Go/No-Go Decision

### ✅ GO - Proceed to Full Training
- Smoke test F1 > 0.30
- No errors
- Training completed in ~2 minutes

**Next**: Set `SMOKE_TEST = False` → Run all cells

---

### ❌ NO-GO - Fix Issues First

**Common issues**:

1. **F1 = 0 or very low**
   - Check model: Must use BioBERT, not bert-base
   - File: Cell 3, `model_name = 'dmis-lab/biobert-v1.1'`

2. **FileNotFoundError: data/...**
   - Wrong pickle path
   - Should be: `data/pickle/{dataset}.pkl`

3. **TypeError: UniversalMedicalDataset**
   - Wrong parameters
   - Use: `task_name=primary_dataset` (not task_type!)

4. **Training never starts**
   - Missing Cell 10 (trainer.train())
   - Re-upload KAGGLE_COMPLETE.ipynb

---

## 📋 Full Training Workflow

### Phase 1: Single Task Validation (30 min)
```python
SMOKE_TEST = False
CONFIG['datasets'] = ['bc2gm']
CONFIG['num_epochs'] = 3  # Quick test
```
**Expected**: F1 = 0.75-0.84

---

### Phase 2: Test All Task Types (2 hours)
```python
# Test one from each type
['bc2gm']      # NER: F1 > 0.80
['chemprot']   # RE: F1 > 0.70
['gad']        # Classification: F1 > 0.80
['pubmedqa']   # QA: F1 > 0.60
['biosses']    # Similarity: Pearson > 0.75
```

---

### Phase 3: Full Matrix (Ready!)
- 7 models × 8 tasks = 56 experiments
- Time: ~3 hours per experiment on T4
- Cost: Free on Kaggle (30h/week) or $0.143/hr on A100

---

## 🚨 Emergency Quick Fixes

### Issue: F1 stuck at 0.46
```python
# Cell 3: Change this line
CONFIG['model_name'] = 'dmis-lab/biobert-v1.1'  # NOT bert-base-uncased!
```

### Issue: FileNotFoundError
```python
# Cell 7: Check this line
pickle_file = Path('data/pickle') / f'{primary_dataset}.pkl'
```

### Issue: TypeError in dataset creation
```python
# Cell 7: Use this
train_dataset = UniversalMedicalDataset(
    data=raw_data['train'],
    tokenizer=tokenizer,
    task_name=primary_dataset,  # Not task_type!
    max_length=CONFIG['max_length']
)
```

---

## 📊 Success Criteria

| Phase | Metric | Expected | Time |
|-------|--------|----------|------|
| Smoke test | F1 | > 0.30 | 2 min |
| Single task (3 epochs) | F1 | > 0.75 | 30 min |
| Single task (10 epochs) | F1 | 0.80-0.84 | 3 hrs |

---

## 📁 Critical Files

### Use These:
- ✅ `KAGGLE_COMPLETE.ipynb` - Complete notebook with all cells
- ✅ `COMPLETE_FIXED_DATASET.py` - Auto-loaded by notebook
- ✅ `COMPLETE_FIXED_MODEL.py` - Auto-loaded by notebook
- ✅ `COMPLETE_FIXED_METRICS.py` - Auto-loaded by notebook

### Don't Use:
- ❌ `KAGGLE_UNIVERSAL.ipynb` - Old, incomplete
- ❌ `KAGGLE_FIXED.ipynb` - Old, missing cells
- ❌ Manual notebook creation - Use generator

---

## 🔧 Validation Commands

```bash
# Before upload
python validate_notebook.py KAGGLE_COMPLETE.ipynb

# Quick data check
python test_pickle_load.py

# Full data test
python tests/test_data_loading.py

# Test tokenization
python tests/test_tokenization.py

# Verify all datasets
python verify_all_datasets.py
```

---

## 💡 Key Reminders

1. **Always use medical models** (BioBERT, not BERT!)
2. **Always run smoke test first** (2 min vs 3 hrs wasted)
3. **Always validate notebook** before upload
4. **Always check file paths** (data/pickle/...)
5. **Always enable checkpointing** (session can die!)

---

## 📖 Full Documentation

For detailed troubleshooting: [TROUBLESHOOTING_GUIDE.md](TROUBLESHOOTING_GUIDE.md)

**Covers**:
- All 13 failures encountered
- Root cause for each
- Complete fixes with code
- Prevention strategies
- Unit tests
- Common pitfalls

---

**Last Updated**: 2026-02-08
**Next Review**: Before Phase 2 (Single-Task baselines)

---

## ✅ Final Check

Before starting experiment:
- [ ] Ran test_pickle_load.py ✅
- [ ] Validated notebook ✅
- [ ] Uploaded to Kaggle ✅
- [ ] Smoke test passed (F1 > 0.30) ✅
- [ ] Ready for full training!

**If all checked → GO FOR LAUNCH!** 🚀
