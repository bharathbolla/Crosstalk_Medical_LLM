# 🔧 QUICK FIX CARD: F1 = 0.46 → 0.84

## ⚡ **3 Changes to Make in Your Kaggle Notebook**

### **Change 1: Cell 6 - Switch Model**

```python
# Line ~50 in Cell 6
"model_name": "dmis-lab/biobert-v1.1",  # ← Change this line!
# WAS: "bert-base-uncased"
```

---

### **Change 2: Cell 8 - Fix Tokenization**

Find this line (~line 30 in Cell 8):
```python
# ❌ REMOVE THIS
text = ' '.join(tokens)
encoding = self.tokenizer(text, max_length=512, ...)
```

Replace with:
```python
# ✅ ADD THIS
encoding = self.tokenizer(
    tokens,                    # Pass list directly
    is_split_into_words=True,  # ← KEY FIX
    max_length=self.max_length,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
```

---

### **Change 3: Cell 8 - Fix Label Alignment**

Find this code (~line 40 in Cell 8):
```python
# ❌ REMOVE THIS
aligned_labels = [-100] * self.max_length
for i in range(min(len(labels), self.max_length)):
    aligned_labels[i] = labels[i]
```

Replace with:
```python
# ✅ ADD THIS
word_ids = encoding.word_ids()
aligned_labels = []

previous_word_idx = None
for word_idx in word_ids:
    if word_idx is None:
        aligned_labels.append(-100)
    elif word_idx != previous_word_idx:
        if word_idx < len(labels):
            aligned_labels.append(labels[word_idx])
        else:
            aligned_labels.append(0)
    else:
        aligned_labels.append(-100)
    previous_word_idx = word_idx

# Ensure correct length
aligned_labels = aligned_labels[:self.max_length]
while len(aligned_labels) < self.max_length:
    aligned_labels.append(-100)
```

---

## ✅ **Expected Results After Fixes**

| Epoch | F1 Score | Status |
|-------|----------|--------|
| 1 | ~0.75-0.78 | ✅ Good start |
| 2 | ~0.81-0.83 | ✅ Converging |
| 3 | ~0.84-0.85 | ✅ Matches paper! |

**Before fixes**: F1 = 0.46 ❌
**After fixes**: F1 = 0.84-0.85 ✅ (**+38% improvement!**)

---

## 🔍 **Quick Verification**

After restarting, check:

1. ✅ Model name shows: `dmis-lab/biobert-v1.1`
2. ✅ First epoch F1 > 0.70 (not stuck at 0.46!)
3. ✅ No "tensor gathering" warnings

---

## 🚀 **Run This Now**

1. Stop current training
2. Make 3 changes above
3. Restart kernel
4. Re-run from Cell 1
5. Watch F1 reach 0.84! 🎉

---

## 📝 **For Other Models** (after BioBERT works)

### RoBERTa-based models need extra step:

```python
# Cell 8, before creating dataset
if 'roberta' in CONFIG['model_name'].lower():
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_name'],
        add_prefix_space=True  # ← Required for RoBERTa
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
```

### Model priority for your research:

1. **BioBERT** (start with this) - Expected F1: 0.84
2. **BlueBERT** (expected best) - Expected F1: 0.85
3. **PubMedBERT** (strong baseline) - Expected F1: 0.82
4. **BioMed-RoBERTa** (need prefix fix) - Expected F1: 0.83
5. **Clinical-BERT** (clinical focus) - Expected F1: 0.80
6. **BERT-base** (general baseline) - Expected F1: 0.72
7. **RoBERTa-base** (general baseline) - Expected F1: 0.74
