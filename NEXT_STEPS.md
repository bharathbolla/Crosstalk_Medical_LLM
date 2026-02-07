# Next Steps - Quick Start Guide

**Status**: ✅ All 39 Python modules implemented and verified
**Date**: 2026-02-07

---

## ⚡ Immediate Actions (Today)

### 1. Install Dependencies

```bash
cd h:\Projects\Research\Cross_Talk_Medical_LLM
pip install -e .
```

This installs all required packages from [requirements.txt](requirements.txt):
- PyTorch ≥2.2.0
- Transformers ≥4.40.0
- PEFT ≥0.10.0
- And 20+ other dependencies

### 2. Verify Installation

```bash
python verify_installation.py
python verify_imports.py
```

Expected output: All modules successfully imported ✓

### 3. Run Unit Tests (Synthetic Data)

```bash
python -m pytest tests/test_synthetic_data.py -v
```

These tests validate core functionality without requiring:
- Real medical datasets
- Model weights
- GPU resources

**Tests included**:
- `TestTokenTracker` — Token counting (RQ5 critical)
- `TestStatisticalFunctions` — Bootstrap CI, permutation tests
- `TestCalibration` — ECE computation
- `TestTransferAnalysis` — Negative transfer detection
- `TestLossFunctions` — Multi-task loss functions

---

## 📝 Apply for PhysioNet Access (DO NOT DELAY)

**⚠️ CRITICAL**: This is a 1-2 week approval process. Start immediately!

### Steps:

1. **Go to**: https://physionet.org/
2. **Complete CITI Training**:
   - Required for credentialed access
   - Takes ~2-3 hours
3. **Submit Credentialed Access Request**:
   - Requires institutional affiliation
   - Need supervisor verification
4. **Wait for Approval**: 1-2 weeks

### Datasets Needed:

All 5 SemEval tasks require PhysioNet access:
- SemEval 2014 Task 7 (Clinical Text Analysis)
- SemEval 2015 Task 14 (Disorder Identification)
- SemEval 2016 Task 12 (Clinical TempEval)
- SemEval 2017 Task 3 (Medical QA)
- SemEval 2021 Task 6 (Clinical NER+RE)

---

## 📊 After Dependencies Installed

### 1. Test Integration

```bash
# Test seqeval for NER metrics
python -c "from seqeval.metrics import f1_score; print('✓ seqeval working')"

# Test sklearn for classification
python -c "from sklearn.metrics import precision_recall_fscore_support; print('✓ sklearn working')"

# Test matplotlib for plots
python -c "import matplotlib.pyplot as plt; print('✓ matplotlib working')"
```

### 2. Run Demo Scripts

```bash
# Verify parameter parity for ablations
python demo/show_ablation_params_llama32.py
```

Expected output: A1-A4 all within ±5% of mean trainable parameters

---

## 🔬 After PhysioNet Access Granted

### Phase 0: Task Parsers (Week 1)

Fill TODOs in all 5 task parsers:

1. **[src/data/semeval2014t7.py](src/data/semeval2014t7.py)**
   - Parse BRAT format annotations
   - Extract disorder entities

2. **[src/data/semeval2015t14.py](src/data/semeval2015t14.py)**
   - Handle discontiguous spans
   - Link span fragments

3. **[src/data/semeval2016t12.py](src/data/semeval2016t12.py)**
   - Parse TimeML XML
   - Extract temporal relations

4. **[src/data/semeval2017t3.py](src/data/semeval2017t3.py)**
   - Parse XML threads
   - Extract QA pairs with relevance labels

5. **[src/data/semeval2021t6.py](src/data/semeval2021t6.py)**
   - Parse both Level 1 (NER) and Level 2 (RE)
   - Extract medication entities and relations

### Phase 0: Contamination Check (Week 2)

```bash
python scripts/run_contamination_check.py \
  --models phi-3-mini gemma-2-2b llama-3.2-3b \
  --tasks all \
  --output results/contamination_report.json
```

This runs the 3-layer contamination protocol:
1. Zero-shot audit
2. N-gram overlap
3. Min-K% probing

**Decision point**: Replace tasks if contamination detected

### Phase 0: Smoke Test (Week 3)

```bash
python scripts/run_smoke_test.py \
  --model phi-3-mini \
  --task semeval2014t7 \
  --steps 50 \
  --samples 100
```

Validates full pipeline before expensive runs:
- Model loading
- Data collation
- Training loop
- Checkpointing
- VRAM monitoring

---

## 🚀 Phase 1: Baselines (Week 4)

### BERT-based Baselines

```bash
# Run all 5 tasks with BERT
python scripts/run_baseline.py \
  --model bert-base-uncased \
  --tasks all \
  --output results/bert_baseline.json

# BioBERT
python scripts/run_baseline.py \
  --model dmis-lab/biobert-base-cased-v1.1 \
  --tasks all \
  --output results/biobert_baseline.json

# ClinicalBERT
python scripts/run_baseline.py \
  --model emilyalsentzer/Bio_ClinicalBERT \
  --tasks all \
  --output results/clinicalbert_baseline.json
```

**Target**: Reproduce published SOTA within 2 F1 points

---

## 🧪 Phase 2-5: Main Experiments (Weeks 5-16)

See [CLAUDE.md](CLAUDE.md) for detailed execution order.

**Key experiments**:
- Phase 2: Single-task (S1) with token counting
- Phase 3: Multi-task (S2, S3a, S3b, S4)
- Phase 4: Quantization (S5) and efficiency
- Phase 5: Analysis and paper writing

---

## 📁 File Locations

### Data Parsers (TODO markers)
- [src/data/semeval2014t7.py](src/data/semeval2014t7.py:52)
- [src/data/semeval2015t14.py](src/data/semeval2015t14.py:53)
- [src/data/semeval2016t12.py](src/data/semeval2016t12.py:54)
- [src/data/semeval2017t3.py](src/data/semeval2017t3.py:55)
- [src/data/semeval2021t6.py](src/data/semeval2021t6.py:73)

### Probing (TODO markers)
- [src/evaluation/probing.py](src/evaluation/probing.py:70)

### Error Analysis (TODO markers)
- [src/evaluation/error_analysis.py](src/evaluation/error_analysis.py:47)

### Contamination (TODO markers)
- [src/evaluation/contamination.py](src/evaluation/contamination.py:56)

---

## 🐛 Known Issues / TODOs

### High Priority (Blocks Progress)
- [ ] Implement task parsers (5 files) — **BLOCKS EVERYTHING**
- [ ] Implement PEFT adapter integration in SharedPrivateLoRA/HierarchicalAdapter
- [ ] Collect probe datasets (UMLS, NegEx, CASI, TimeML)

### Medium Priority
- [ ] Write integration tests with mocked BIO data
- [ ] Create experiment config YAMLs in `configs/`
- [ ] Write execution scripts in `scripts/`

### Low Priority
- [ ] Implement quantization modules (`src/quantization/`)
- [ ] Create analysis notebooks (8 planned)
- [ ] Set up CI/CD for automated testing

---

## 💰 Budget & Resources

**Primary Platform**: Kaggle free T4 GPU (16 GB VRAM)
- QLoRA 4-bit for 7-8B models
- FP16 LoRA for 2-3B models
- Checkpoint every 200 steps

**Optional Budget**: $0-$80 for L4/A100 time
- Use only if T4 insufficient
- Monitor costs carefully

---

## 📚 Documentation

### Main Documents
- [CLAUDE.md](CLAUDE.md) — Master orchestration
- [TRAINING_EVALUATION_SUMMARY.md](TRAINING_EVALUATION_SUMMARY.md) — Detailed evaluation docs
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) — Module inventory
- [README.md](README.md) — Project overview

### Code Documentation
- All modules have comprehensive docstrings
- Type hints throughout
- TODOs marked where real data needed

---

## ✅ Verification Checklist

Before starting experiments, verify:

- [ ] Dependencies installed (`pip install -e .`)
- [ ] All imports working (`python verify_imports.py`)
- [ ] Unit tests passing (`pytest tests/test_synthetic_data.py`)
- [ ] PhysioNet access granted
- [ ] Task parsers implemented
- [ ] Contamination check completed
- [ ] Smoke test passed on one task

---

## 🎯 Critical Success Factors

### RQ5: Token-Controlled Baseline (MOST IMPORTANT)
- `TokenTracker` logs every token at every step
- `TokenControlledTrainer` stops at exact token count
- Compare multi-task vs single-task with token parity

### RQ4: Negative Transfer Detection
- `PCGradOptimizer` tracks gradient conflicts
- `detect_negative_transfer()` identifies problematic tasks
- Transfer matrix analysis

### RQ1-RQ3: Performance Analysis
- Bootstrap confidence intervals (10K samples)
- Paired permutation tests (10K permutations)
- Win/tie/loss counts per task

---

**Last Updated**: 2026-02-07
**Status**: Ready for Phase 0 execution after dependency installation
