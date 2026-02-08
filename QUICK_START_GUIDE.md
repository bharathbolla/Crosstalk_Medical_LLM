# Quick Start Guide
## From Zero to Publication-Ready Results in 3 Days

## 📋 Overview

| Phase | Platform | Time | Cost | Purpose |
|-------|----------|------|------|---------|
| **Phase 1** | Kaggle Free | 2-3h | $0 | Smoke test all 56 experiments |
| **Phase 2** | vast.ai A100 | 40h (2 days) | $5.72 | Full runs for paper |
| **Total** | - | **2 days** | **$5.72** | Complete results |

---

## 🚀 Phase 1: Kaggle Free - Smoke Tests (TODAY)

### Goal
✅ Verify all 7 models × 8 datasets work without errors
✅ Catch bugs before paying for A100
✅ Total time: 2-3 hours
✅ Total cost: $0

### Step 1.1: Upload to Kaggle (5 min)

1. Go to https://www.kaggle.com/code
2. Click "New Notebook"
3. Upload `KAGGLE_RESEARCH_EXPERIMENTS.ipynb`
4. Settings:
   - GPU: T4 x2
   - Internet: ON
   - Persistence: Enable

### Step 1.2: Setup (10 min)

```python
# Run Cell 1: Clone repo
# Run Cell 2: Install dependencies
# Run Cell 3: Configuration
```

Verify output shows:
```
✅ PyTorch: 2.x
✅ CUDA: True
✅ GPU: Tesla T4
✅ VRAM: 16.0 GB
```

### Step 1.3: Quick Test (5 min)

Update Cell 3 with minimal config:
```python
CONFIG = {
    'max_samples_per_dataset': 50,  # Quick test
    'num_epochs': 1,
    'batch_size': 16,
    'datasets': ['bc2gm'],  # Just one dataset
    'model_name': 'bert-base-uncased',  # Just one model
}
```

Run Cells 4-7. Should complete in ~2 minutes.

✅ If this works → proceed to Step 1.4
❌ If this fails → fix errors first

### Step 1.4: Full Smoke Test (2-3 hours)

Run all 56 combinations:

```python
models = [
    'bert-base-uncased',
    'roberta-base',
    'dmis-lab/biobert-v1.1',
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'emilyalsentzer/Bio_ClinicalBERT',
    'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
    'allenai/biomed_roberta_base',
]

datasets = [
    'bc2gm', 'jnlpba', 'chemprot', 'ddi',
    'gad', 'hoc', 'pubmedqa', 'biosses'
]

CONFIG['max_samples_per_dataset'] = 50
CONFIG['num_epochs'] = 1

for model in models:
    for dataset in datasets:
        CONFIG['model_name'] = model
        CONFIG['datasets'] = [dataset]
        # Run Cells 4-7
        # Results auto-save to CSV
```

### Step 1.5: Verify Results (5 min)

```python
import pandas as pd

df = pd.read_csv('results/all_experiments.csv')

print(f"Total experiments: {len(df)}")  # Should be 56
print(f"Success: {len(df[df['test_f1'] > 0])}")  # Should be 56
print(f"Average F1: {df['test_f1'].mean():.3f}")  # Should be > 0.3

# Check for errors
assert len(df) == 56, "Missing experiments!"
assert df['test_f1'].min() > 0, "Some experiments failed!"

print("✅ ALL SMOKE TESTS PASSED!")
```

### Step 1.6: Download Results (5 min)

1. Click "Save Version"
2. Wait for notebook to finish
3. Go to "Data" tab
4. Download `results/all_experiments.csv`
5. Save locally for backup

---

## 🔥 Phase 2: vast.ai A100 - Full Runs (2 DAYS)

### Goal
✅ Run all 56 experiments with full data
✅ Publication-quality results
✅ Total time: 40 GPU hours (2 days continuous)
✅ Total cost: $5.72 @ $0.143/hr

### Step 2.1: Rent A100 (10 min)

1. Go to https://vast.ai
2. Create account (if needed)
3. Add $10 credit
4. Search filters:
   - GPU: A100-SXM4-40GB
   - VRAM: >38 GB
   - Type: Interruptible
   - Reliability: >98%
5. Sort by price (low to high)
6. Find: ~$0.143/hr
7. Click "Rent"

### Step 2.2: Connect to Instance (5 min)

```bash
# SSH (shown on vast.ai dashboard)
ssh -p 12345 root@123.45.67.89 -L 8080:localhost:8080

# Or use Jupyter (easier)
# Click "Open" button on vast.ai dashboard
```

### Step 2.3: Setup Instance (10 min)

```bash
# Clone your repo
git clone https://github.com/bharathbolla/Crosstalk_Medical_LLM.git
cd Crosstalk_Medical_LLM

# Install dependencies
pip install transformers torch accelerate scikit-learn seqeval pandas

# Verify GPU
nvidia-smi
# Should show: A100-SXM4-40GB

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should show: CUDA: True
```

### Step 2.4: Upload Data (5 min)

```bash
# From your local machine, upload pickle data
scp -P 12345 -r data/pickle/* root@123.45.67.89:/root/Crosstalk_Medical_LLM/data/pickle/

# Or re-run test_pickle_load.py on instance
python test_pickle_load.py
```

### Step 2.5: Start Full Runs (30 sec)

Open Jupyter or run:

```bash
# Start Jupyter
jupyter notebook --no-browser --port=8080 --ip=0.0.0.0 --allow-root

# Then access via: localhost:8080 on your browser
```

In notebook, update Cell 3:

```python
CONFIG = {
    'max_samples_per_dataset': None,  # ALL data
    'num_epochs': 10,  # Early stopping will stop earlier
    'batch_size': 64,  # A100 optimal
    'use_early_stopping': True,
    'early_stopping_patience': 3,
    'save_steps': 100,  # Checkpoint every ~2 min
    'resume_from_checkpoint': True,
}
```

Queue all 56 experiments:

```python
for model in models:
    for dataset in datasets:
        CONFIG['model_name'] = model
        CONFIG['datasets'] = [dataset]
        # Run Cells 4-7
        # Auto-saves to CSV
        # Checkpoints every 100 steps
```

### Step 2.6: Monitor Progress (Optional)

```bash
# Check progress
tail -f results/all_experiments.csv

# Check GPU usage
watch -n 5 nvidia-smi

# Count completed experiments
wc -l results/all_experiments.csv
# Should increment: 1, 2, 3... up to 56
```

### Step 2.7: Handle Interruptions (Auto)

If instance gets interrupted:
- Notebook auto-resumes from last checkpoint (every ~2 min)
- Only lose ~2 minutes of work
- CSV results preserved
- Re-run same cell to continue

### Step 2.8: Download Final Results (5 min)

After all 56 experiments complete:

```bash
# From your local machine
scp -P 12345 -r root@123.45.67.89:/root/Crosstalk_Medical_LLM/results ./results_final

# Should have:
# - all_experiments.csv (56 rows)
# - results_*.json (56 files)
# - results_*.csv (56 files)
```

### Step 2.9: Stop Instance

1. Go to vast.ai dashboard
2. Click "Destroy" instance
3. Verify no charges accruing

**Total cost**: Check dashboard (should be ~$5.72)

---

## 📊 Phase 3: Analysis (LOCAL)

### Step 3.1: Load Results (2 min)

```python
import pandas as pd

df = pd.read_csv('results_final/all_experiments.csv')

print(f"Total experiments: {len(df)}")  # Should be 56
print(f"Unique models: {df['model_name'].nunique()}")  # Should be 7
print(f"Unique datasets: {df['dataset'].nunique()}")  # Should be 8
```

### Step 3.2: Generate Tables (5 min)

```bash
python compare_experiments.py results_final/all_experiments.csv
```

Outputs:
- Model comparison table
- Dataset comparison table
- Top 5 experiments by F1
- Token efficiency analysis
- LaTeX table → `results/latex_table.tex`

### Step 3.3: Identify Top 3 Models (2 min)

```python
# Average F1 by model
model_avg = df.groupby('model_name')['test_f1'].mean().sort_values(ascending=False)
print(model_avg)

# Top 3 models
top_3 = model_avg.head(3).index.tolist()
print(f"\nTop 3 models for Phase 4 (multi-task):")
for i, model in enumerate(top_3, 1):
    print(f"{i}. {model}: {model_avg[model]:.4f}")
```

Expected output:
```
Top 3 models:
1. BlueBERT: 0.8124
2. PubMedBERT: 0.7956
3. Clinical-BERT: 0.7834
```

---

## 📅 Complete Timeline

### Day 1 (Today)
- **Morning** (2h): Kaggle smoke tests
- **Afternoon** (2h): Rent A100, setup, start runs
- **Evening**: Let A100 run overnight

### Day 2 (Tomorrow)
- **All day**: A100 runs continue
- **Evening**: Download results, stop instance

### Day 3 (Optional)
- **Morning** (2h): Analysis, generate tables
- **Afternoon**: Start writing paper!

---

## ✅ Success Criteria

### After Phase 1 (Smoke Tests)
- [x] All 56 smoke tests pass
- [x] CSV has 56 rows
- [x] All F1 scores > 0.3 (even with 50 samples)
- [x] No errors in console

### After Phase 2 (Full Runs)
- [x] All 56 full experiments complete
- [x] Average F1 > 0.70
- [x] Best model F1 > 0.80
- [x] CSV properly formatted
- [x] Total cost < $7

### After Phase 3 (Analysis)
- [x] Top 3 models identified
- [x] LaTeX table generated
- [x] Ready for paper writing
- [x] Ready for Phase 4 (multi-task)

---

## 🚨 Troubleshooting

### Kaggle Smoke Test Fails
**Problem**: Experiment fails with error
**Solution**: Fix error, re-run just that combination

### A100 Instance Interrupted
**Problem**: Instance taken away mid-training
**Solution**: Auto-resumes from checkpoint (built-in), re-run cell

### Out of Memory on A100
**Problem**: CUDA OOM error
**Solution**: Reduce batch size to 48, re-run

### CSV Missing Data
**Problem**: all_experiments.csv has < 56 rows
**Solution**: Re-run missing experiments, results will append

---

## 💡 Pro Tips

1. **Start on a Monday**: Gives you full week for testing
2. **Run smoke tests first thing**: Catch errors early
3. **Download CSV frequently**: Every 10 experiments
4. **Keep instance monitored**: Check every few hours
5. **Backup results**: Download to multiple locations

---

## 📞 Next Steps

After completing all 56 single-task experiments:

1. ✅ Analyze results → identify top 3 models
2. ✅ Run multi-task experiments (Phase 4)
   - 4 combinations × 3 models = 12 experiments
   - Cost: ~$2
   - Time: ~15 hours
3. ✅ Run token-controlled (Phase 5)
   - Top 3 models only
   - Cost: ~$10
   - Time: ~70 hours
4. ✅ Write paper! 🎉

---

**Total Investment for ALL phases**:
- Time: ~1 week calendar time
- Cost: ~$18 total
- Result: Publication-ready results for BioNLP Workshop/EMNLP Findings!

**Good luck! 🚀**
