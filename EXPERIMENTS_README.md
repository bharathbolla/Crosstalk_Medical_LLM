# Research Experiments - Quick Start

## 🎯 You Now Have Everything for Your Paper!

### What's Ready:

1. ✅ **Data**: 8 datasets in pickle format (35MB, zero dependencies)
2. ✅ **Experiment Notebook**: Full research pipeline with tracking
3. ✅ **Analysis Scripts**: Automated result processing
4. ✅ **Documentation**: Complete experiment guide

---

## 🚀 3-Step Workflow

### Step 1: Run Experiments on Kaggle

1. **Upload Notebook**:
   - Open Kaggle
   - Create new notebook
   - Upload `KAGGLE_RESEARCH_EXPERIMENTS.ipynb`
   - Settings: GPU T4 x2 + Internet ON

2. **Configure Experiment** (Cell 3):
   ```python
   CONFIG = {
       "experiment_type": "single_task_baseline",
       "datasets": ["bc2gm"],
       "model_name": "dmis-lab/biobert-v1.1",
       "num_epochs": 3,
       "track_tokens": True,  # CRITICAL for RQ5!
   }
   ```

3. **Run All Cells**:
   - Cells 1-2: Setup
   - Cell 3: Configure
   - Cells 4-8: Train & evaluate
   - Results saved automatically!

### Step 2: Download Results

After session ends:
1. Kaggle → Output → Create Dataset
2. Include `results/` and `models/` directories
3. Download to your local machine

### Step 3: Analyze for Paper

On your local machine:

```bash
# Install analysis dependencies (once)
pip install pandas numpy scipy matplotlib

# Analyze all results
python analyze_results.py --results_dir results/ --output_dir analysis/

# This generates:
# - all_results.csv (all data)
# - table_*.tex (LaTeX tables for paper)
# - statistics.json (p-values, effect sizes)
# - SUMMARY.txt (human-readable report)
```

---

## 📊 Key Files

### For Running Experiments:
- `KAGGLE_RESEARCH_EXPERIMENTS.ipynb` - Main experiment notebook
- `EXPERIMENT_GUIDE.md` - Detailed experiment configurations
- `data/pickle/*.pkl` - All 8 datasets

### For Analysis:
- `analyze_results.py` - Automated result analysis
- `notebooks/` - Visualization notebooks (optional)

### Documentation:
- `CLAUDE.md` - Master research plan
- `MEMORY.md` - Implementation notes
- `EXPERIMENTS_README.md` - This file!

---

## 🔬 Research Questions (RQs)

Your experiments answer these questions:

### RQ1: Does Multi-Task Help?
**Experiments**: Single-task (S1) vs Multi-task (S2)
- Run single-task on each dataset
- Run multi-task on combined datasets
- Compare F1 scores
- **Analysis**: `table_multitask_comparison.tex`

### RQ2: Out-of-Distribution Transfer?
**Experiments**: Train on A+B, test on C
- Multi-task on subset of datasets
- Evaluate on held-out dataset
- **Analysis**: Custom evaluation in Cell 9

### RQ3: Model Size vs Performance?
**Experiments**: Vary model sizes
- bert-tiny (4M params)
- bert-base (110M params)
- biobert (110M params)
- **Analysis**: `all_results.csv` → model comparison

### RQ4: When Does Negative Transfer Occur?
**Experiments**: Task similarity analysis
- Try different task combinations
- Measure when multi-task < single-task
- **Analysis**: Statistical tests in `statistics.json`

### RQ5: Real Transfer vs Data Exposure? ⭐ CRITICAL
**Experiments**: Token-controlled baseline
1. Run multi-task (tracks tokens)
2. Run single-task with SAME token count
3. If multi-task still better → genuine transfer!
- **Analysis**: `table_rq5_token_controlled.tex`

---

## 📝 Experiment Checklist

### Week 1: Single-Task Baselines
- [ ] bc2gm - BERT
- [ ] bc2gm - BioBERT
- [ ] bc2gm - PubMedBERT
- [ ] jnlpba - BERT
- [ ] jnlpba - BioBERT
- [ ] jnlpba - PubMedBERT
- [ ] (Repeat for all 8 datasets × 3 models = 24 runs)

### Week 2: Multi-Task Experiments
- [ ] NER tasks: bc2gm + jnlpba
- [ ] RE tasks: chemprot + ddi
- [ ] Mixed: bc2gm + chemprot + gad
- [ ] All 8 datasets together

### Week 3: Token-Controlled (RQ5)
- [ ] Get token counts from multi-task runs
- [ ] Run single-task with matching tokens
- [ ] Compare results (THE CRITICAL TABLE!)

### Week 4: Analysis & Writing
- [ ] Run `analyze_results.py`
- [ ] Generate all tables/figures
- [ ] Write paper sections
- [ ] Statistical tests

---

## 💡 Pro Tips

### Quick Test First
Before long runs, test with:
```python
CONFIG['max_samples_per_dataset'] = 100  # Tiny subset
CONFIG['num_epochs'] = 1
```
Runs in 5 minutes, verifies everything works!

### Save Often
Kaggle sessions can die unexpectedly:
- Notebook automatically checkpoints every 500 steps
- Download results frequently!

### Track Tokens (RQ5)
ALWAYS set:
```python
CONFIG['track_tokens'] = True
```
This is YOUR KEY CONTRIBUTION to the field!

### Use Wandb (Optional)
For experiment tracking:
```python
CONFIG['use_wandb'] = True
CONFIG['wandb_project'] = 'your-project-name'
```
Login: `wandb login` (in Cell 2)

---

## 📈 Expected Results

Based on similar papers:

- **Single-task F1**: 0.75-0.85 (varies by dataset)
- **Multi-task improvement**: +2-5% F1
- **Token-controlled**: Should show genuine transfer (RQ5!)
- **Training time**: 2-3 hours per experiment on T4

---

## 🆘 Troubleshooting

### "Out of Memory"
```python
CONFIG['batch_size'] = 8  # Reduce
CONFIG['max_length'] = 256  # Shorter sequences
```

### "No pickle files found"
- Check: `ls data/pickle/` shows .pkl files
- If missing: Clone repo with Internet ON

### "Results not saving"
- Check: `ls results/` shows JSON files
- Manually copy: `!cp -r results/ /kaggle/working/`

### "Analysis script fails"
- Install: `pip install pandas scipy matplotlib`
- Check: results_dir has `results_*.json` files

---

## 📚 Paper Writing

### Methods Section
"We conducted experiments using our token-controlled baseline methodology
(RQ5), ensuring fair comparison between single-task and multi-task models
by equalizing total training tokens..."

### Results Section
Tables generated by `analyze_results.py`:
- Table 1: Single-task baselines (`table_single_task.tex`)
- Table 2: Multi-task comparison (`table_multitask_comparison.tex`)
- Table 3: Token-controlled analysis (`table_rq5_token_controlled.tex`)

### Statistical Tests
From `statistics.json`:
- "Multi-task significantly outperformed single-task (t=X.XX, p<0.05)"
- "Effect size was large (Cohen's d=X.XX)"
- "Even under token parity, multi-task showed +X% improvement"

---

## 🎉 You're Ready!

Everything is set up. Just:
1. Open Kaggle
2. Upload notebook
3. Run experiments
4. Download results
5. Analyze
6. Write paper

**Questions?** Check `EXPERIMENT_GUIDE.md` for detailed configurations.

**Good luck with your research! 🚀**
