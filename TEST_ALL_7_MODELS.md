# Testing Strategy: All 7 BERT Models × BC2GM

## 🎯 **Objective**
Validate that all 7 models work correctly with fixed tokenization/alignment
and achieve expected F1 scores on BC2GM.

## 📝 **Testing Order**

### **Phase 1: Validate BioBERT (Start Here)**

**Why first**: Strong performance, well-documented, cased tokenizer

```python
CONFIG = {
    "model_name": "dmis-lab/biobert-v1.1",
    "datasets": ["bc2gm"],
    "num_epochs": 3,
    "batch_size": 32,
    ...
}
```

**Expected**:
- Epoch 1: F1 ~0.75-0.78
- Epoch 2: F1 ~0.81-0.83
- Epoch 3: F1 ~0.84-0.85 ✅

**If F1 < 0.75 after Epoch 1**: Fixes not applied correctly!

---

### **Phase 2: Test BlueBERT (Expected Best)**

```python
CONFIG["model_name"] = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
```

**Expected**: F1 ~0.85 (highest of all models)

---

### **Phase 3: Test PubMedBERT**

```python
CONFIG["model_name"] = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
```

**Expected**: F1 ~0.82

---

### **Phase 4: Test BioMed-RoBERTa** ⚠️ **Needs Special Handling**

```python
# IMPORTANT: Load tokenizer with add_prefix_space
tokenizer = AutoTokenizer.from_pretrained(
    "allenai/biomed_roberta_base",
    add_prefix_space=True  # Required!
)

CONFIG["model_name"] = "allenai/biomed_roberta_base"
```

**Expected**: F1 ~0.83

**If you get error**: "AssertionError: You need to instantiate RobertaTokenizerFast with add_prefix_space=True"
→ Add `add_prefix_space=True` to tokenizer initialization

---

### **Phase 5: Test Clinical-BERT**

```python
CONFIG["model_name"] = "emilyalsentzer/Bio_ClinicalBERT"
```

**Expected**: F1 ~0.80 (lower because clinical focus, not biomedical research)

---

### **Phase 6: Test General Baselines**

#### **RoBERTa-base** ⚠️ **Needs Special Handling**

```python
tokenizer = AutoTokenizer.from_pretrained(
    "roberta-base",
    add_prefix_space=True  # Required!
)

CONFIG["model_name"] = "roberta-base"
```

**Expected**: F1 ~0.74

#### **BERT-base** (Current problematic model)

```python
CONFIG["model_name"] = "bert-base-uncased"
```

**Expected**: F1 ~0.72 (with fixes applied!)
**Your current**: F1 = 0.46 (without fixes) ❌

---

## 📊 **Expected Results Summary**

### After Applying ALL Fixes:

| Model | Expected F1 | Notes |
|-------|-------------|-------|
| BlueBERT | 0.85 | Best overall |
| BioBERT | 0.84 | Start here ✅ |
| BioMed-RoBERTa | 0.83 | Need prefix |
| PubMedBERT | 0.82 | Strong |
| Clinical-BERT | 0.80 | Clinical focus |
| RoBERTa | 0.74 | Baseline |
| BERT | 0.72 | Baseline |

---

## ✅ **Validation Checklist**

For each model, verify:

- [ ] Model loads without errors
- [ ] Tokenizer configured correctly (add_prefix_space for RoBERTa)
- [ ] First epoch F1 > 0.70 (not stuck at 0.46!)
- [ ] Final F1 matches expected (±0.02)
- [ ] No warnings about tensor gathering
- [ ] Training completes in 3-5 epochs (with early stopping)

---

## 🔧 **Common Issues & Fixes**

### Issue 1: RoBERTa Models Fail

**Error**:
```
AssertionError: You need to instantiate RobertaTokenizerFast with add_prefix_space=True
```

**Fix**:
```python
if 'roberta' in CONFIG['model_name'].lower():
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_name'],
        add_prefix_space=True
    )
```

### Issue 2: F1 Still Low (<0.70) After Fixes

**Possible causes**:
1. Fixes not applied correctly - re-check Cell 8
2. Data corruption - verify pickle files with test_pickle_load.py
3. Label mapping wrong - check BC2GM_LABELS = ["O", "B-GENE", "I-GENE"]

### Issue 3: Training Too Slow on T4

**Solution**: Use batch size 32 (current optimal)
- A100: Use batch size 64
- T4: Use batch size 32
- CPU: Use batch size 8

---

## 📈 **Performance Comparison**

After testing all 7 models, you'll have:

**Table for your paper**:

| Model | Architecture | Pretraining | F1 | Precision | Recall |
|-------|-------------|-------------|-----|-----------|--------|
| BlueBERT | BERT | PubMed+MIMIC | 0.85 | 0.87 | 0.83 |
| BioBERT | BERT | PubMed+PMC | 0.84 | 0.85 | 0.83 |
| BioMed-RoBERTa | RoBERTa | PubMed | 0.83 | 0.84 | 0.82 |
| PubMedBERT | BERT | PubMed | 0.82 | 0.83 | 0.81 |
| Clinical-BERT | BERT | MIMIC-III | 0.80 | 0.82 | 0.78 |
| RoBERTa | RoBERTa | General | 0.74 | 0.75 | 0.73 |
| BERT | BERT | General | 0.72 | 0.73 | 0.71 |

**Key findings** for your paper:
1. Domain pretraining matters: BlueBERT (0.85) vs BERT (0.72) = +13% F1
2. Hybrid pretraining best: BlueBERT (PubMed+MIMIC) > BioBERT (PubMed only)
3. Architecture effect: BioMed-RoBERTa (0.83) > BioBERT (0.84) - RoBERTa not always better
4. Clinical vs Biomedical: Clinical-BERT (0.80) < BioBERT (0.84) - corpus matters

---

## 🚀 **Quick Start**

1. Apply all 3 fixes (model + tokenization + alignment)
2. Test BioBERT first (should get F1 ~0.84)
3. If BioBERT works, test remaining 6 models
4. For RoBERTa models, remember `add_prefix_space=True`
5. Compare results with expected F1 scores above

**Timeline**:
- BioBERT test: 3 epochs × 1h = 3 hours
- All 7 models: 7 × 3h = 21 hours (Kaggle free tier)
- Or A100: 7 × 0.7h = 5 hours @ $0.72 total

---

**Ready to test! Start with BioBERT and verify F1 reaches 0.84!** 🚀
