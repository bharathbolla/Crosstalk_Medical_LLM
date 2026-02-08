# CSV Export Guide

## Overview

Every experiment automatically exports results to **CSV format** for easy comparison and analysis. This makes it simple to:
- Compare experiments side-by-side
- Create tables for your research paper
- Track progress across multiple runs
- Identify best-performing models

---

## What Gets Exported

### Experiment Metadata
- `experiment_id` — Unique timestamp ID (e.g., `20260207_143052`)
- `timestamp` — ISO format timestamp
- `experiment_type` — `single_task`, `multi_task`, or `token_controlled`
- `model_name` — HuggingFace model ID
- `dataset` — Primary dataset name

### Training Configuration
- `batch_size` — Batch size used
- `learning_rate` — Learning rate
- `num_epochs_max` — Maximum epochs allowed
- `actual_epochs` — Actual epochs trained (with early stopping)
- `early_stopping` — Whether early stopping was enabled

### Dataset Statistics
- `train_samples` — Number of training samples
- `test_samples` — Number of test samples
- `num_datasets` — Number of datasets (1 for single-task, >1 for multi-task)

### Model Statistics
- `total_params` — Total model parameters
- `trainable_params` — Trainable parameters

### Training Results
- `train_loss` — Final training loss
- `train_runtime_seconds` — Total training time
- `train_samples_per_second` — Training throughput
- `tokens_processed` — **CRITICAL for RQ5** — Total tokens seen

### Test Results
- `test_f1` — F1 score on test set ⭐ **PRIMARY METRIC**
- `test_precision` — Precision score
- `test_recall` — Recall score
- `test_loss` — Test loss

---

## File Locations

### After Each Experiment:

1. **Master CSV** (all experiments combined):
   ```
   results/all_experiments.csv
   ```
   - **Appends** one row per experiment
   - **Accumulates** across all runs
   - **Use for** comparing all experiments

2. **Individual CSV** (single experiment):
   ```
   results/results_{EXPERIMENT_ID}.csv
   ```
   - **One file** per experiment
   - **Same data** as master CSV row
   - **Use for** archiving specific runs

3. **JSON** (detailed results):
   ```
   results/results_{EXPERIMENT_ID}.json
   ```
   - Full experiment configuration
   - Detailed metrics
   - Complete training history

---

## Example CSV Output

```csv
experiment_id,timestamp,experiment_type,model_name,dataset,test_f1,test_precision,test_recall,tokens_processed
20260207_143052,2026-02-07T14:30:52,single_task,bert-base-uncased,bc2gm,0.8234,0.8156,0.8314,2456789
20260207_150234,2026-02-07T15:02:34,single_task,dmis-lab/biobert-v1.1,bc2gm,0.8567,0.8489,0.8647,2456789
20260207_162045,2026-02-07T16:20:45,multi_task,dmis-lab/biobert-v1.1,bc2gm,0.8623,0.8534,0.8714,7234561
```

---

## How to Use CSV Results

### 1. Quick View in Excel/Google Sheets

Simply open `results/all_experiments.csv` in Excel or Google Sheets:
- Sort by `test_f1` to find best models
- Filter by `experiment_type` to compare single vs multi-task
- Create pivot tables for analysis

### 2. Automated Analysis (Recommended)

Use the provided analysis script:

```bash
python compare_experiments.py results/all_experiments.csv
```

**Generates**:
- Model comparison table
- Dataset comparison table
- Single-task vs Multi-task comparison
- Top 5 experiments by F1
- Token efficiency analysis
- Training efficiency analysis
- Early stopping statistics
- LaTeX table for paper

### 3. Python Analysis

```python
import pandas as pd

# Load all experiments
df = pd.read_csv('results/all_experiments.csv')

# Compare models
best_model = df.loc[df['test_f1'].idxmax()]
print(f"Best model: {best_model['model_name']}")
print(f"F1 score: {best_model['test_f1']:.4f}")

# Compare experiment types
comparison = df.groupby('experiment_type')['test_f1'].mean()
print(comparison)

# Token-controlled comparison (RQ5)
st = df[df['experiment_type'] == 'single_task']
mt = df[df['experiment_type'] == 'multi_task']
print(f"Multi-task gain: {mt['test_f1'].mean() - st['test_f1'].mean():.4f}")
```

---

## Common Analysis Tasks

### Task 1: Find Best Model for Each Dataset

```python
best_per_dataset = df.loc[df.groupby('dataset')['test_f1'].idxmax()]
print(best_per_dataset[['dataset', 'model_name', 'test_f1']])
```

### Task 2: Compare BERT Variants

```python
bert_models = df[df['model_name'].str.contains('bert', case=False)]
comparison = bert_models.groupby('model_name')['test_f1'].mean().sort_values(ascending=False)
print(comparison)
```

### Task 3: Token-Controlled Analysis (RQ5)

```python
# Get experiments with same token count
target_tokens = 5000000

# Single-task at target tokens
st_controlled = df[(df['experiment_type'] == 'token_controlled') &
                   (df['tokens_processed'].between(target_tokens * 0.95, target_tokens * 1.05))]

# Multi-task naturally at that token count
mt_normal = df[(df['experiment_type'] == 'multi_task') &
               (df['tokens_processed'].between(target_tokens * 0.95, target_tokens * 1.05))]

print(f"ST (token-controlled) F1: {st_controlled['test_f1'].mean():.4f}")
print(f"MT (natural) F1: {mt_normal['test_f1'].mean():.4f}")
print(f"Genuine transfer gain: {mt_normal['test_f1'].mean() - st_controlled['test_f1'].mean():.4f}")
```

### Task 4: Generate Paper Table

```python
# Create publication-ready table
paper_table = df.groupby(['experiment_type', 'model_name', 'dataset']).agg({
    'test_f1': ['mean', 'std'],
    'test_precision': 'mean',
    'test_recall': 'mean'
}).round(3)

print(paper_table.to_latex())
```

---

## Tips

### For Running Multiple Experiments

1. **Never delete** `results/all_experiments.csv` — it accumulates all runs!
2. **Backup regularly** — copy to local machine after each Kaggle session
3. **Check duplicates** — if re-running, filter by timestamp

### For Paper Writing

1. **Run comparison script** after completing all experiments:
   ```bash
   python compare_experiments.py
   ```

2. **Copy LaTeX table** from `results/latex_table.tex` to paper

3. **Create figures** using pandas/matplotlib:
   ```python
   import matplotlib.pyplot as plt

   df.groupby('model_name')['test_f1'].mean().plot(kind='bar')
   plt.ylabel('F1 Score')
   plt.title('Model Comparison')
   plt.savefig('figures/model_comparison.png')
   ```

### For Token-Controlled Analysis (RQ5)

**This is your PRIMARY CONTRIBUTION!**

1. Run single-task experiments → note token counts
2. Run multi-task experiments → note token counts
3. Run token-controlled single-task with same total tokens
4. Compare in CSV:
   ```python
   # Example: BC2GM dataset
   mt_tokens = df[(df['experiment_type'] == 'multi_task') &
                  (df['dataset'] == 'bc2gm')]['tokens_processed'].iloc[0]

   st_controlled = df[(df['experiment_type'] == 'token_controlled') &
                      (df['dataset'] == 'bc2gm') &
                      (df['tokens_processed'] >= mt_tokens * 0.95)]
   ```

---

## Troubleshooting

### CSV Not Created

**Problem**: No `all_experiments.csv` file

**Solution**:
- Make sure you ran Cell 7 in the notebook
- Check that `results/` directory exists
- Verify pandas is installed

### Duplicate Rows

**Problem**: Same experiment appears multiple times

**Solution**:
```python
# Remove duplicates (keep first occurrence)
df = pd.read_csv('results/all_experiments.csv')
df = df.drop_duplicates(subset=['experiment_id'], keep='first')
df.to_csv('results/all_experiments.csv', index=False)
```

### Missing Columns

**Problem**: Some columns don't exist in older runs

**Solution**:
```python
# Fill missing columns with defaults
df = pd.read_csv('results/all_experiments.csv')
df['tokens_processed'] = df['tokens_processed'].fillna(0)
df.to_csv('results/all_experiments.csv', index=False)
```

---

## Summary

✅ **Every experiment** automatically saves to CSV
✅ **Master CSV** accumulates all experiments for easy comparison
✅ **Individual CSV** archives each run separately
✅ **JSON** preserves full configuration details
✅ **Analysis script** generates publication-ready tables
✅ **Token tracking** enables rigorous RQ5 analysis

**Next**: Run experiments and use `python compare_experiments.py` for instant analysis!
