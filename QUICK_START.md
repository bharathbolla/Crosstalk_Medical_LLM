# Quick Start Guide - Ready in 2 Minutes! 🚀

**Last Updated**: 2026-02-07
**Status**: ✅ All datasets included - no downloads needed!

---

## 🎯 On Kaggle (Recommended)

### Step 1: Create Notebook
1. Go to Kaggle.com
2. Create new notebook  
3. Settings → **GPU T4 x2** + **Internet ON**

### Step 2: Clone Repo (Datasets Included!)
```python
!git clone https://github.com/bharathbolla/Crosstalk_Medical_LLM.git
%cd Crosstalk_Medical_LLM
```

### Step 3: Install Dependencies
```python
!pip install -q transformers datasets evaluate wandb accelerate scikit-learn pyyaml
```

### Step 4: Verify Everything Works
```python
!python test_parsers.py
```

Expected: `[SUCCESS] All 8 parsers work!`

### Step 5: Start Training!
```python
!python scripts/run_baseline.py --model bert-base-uncased --task bc2gm --epochs 3 --batch_size 16
```

**That's it! From zero to training in 2 minutes!** 🎉

---

## 📦 What You Get

8 datasets (49,591 samples) pre-included:
- bc2gm (12,574) - Gene/protein NER
- jnlpba (18,607) - Bio-entity NER  
- chemprot (1,020) - Chemical-protein RE
- ddi (571) - Drug-drug RE
- gad (3,836) - Gene-disease classification
- hoc (12,119) - Cancer hallmarks
- pubmedqa (800) - Medical QA
- biosses (64) - Sentence similarity

**No downloads. No errors. Just works!** ✅
