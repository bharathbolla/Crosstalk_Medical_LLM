# Research Experiment Guide

## Quick Start: Running Experiments on Kaggle

### 1. Setup (One-Time)
1. Create new Kaggle notebook
2. Settings: **GPU T4 x2 + Internet ON**
3. Upload `KAGGLE_RESEARCH_EXPERIMENTS.ipynb`
4. Run Cells 1-2 (setup and install)

### 2. Configure Experiment (Cell 3)

Edit the `CONFIG` dictionary based on your experiment type:

## Experiment Types for Paper

### A. Single-Task Baselines (S1) - Week 5-7

**Purpose**: Baseline for each dataset individually

```python
CONFIG = {
    "experiment_type": "single_task_baseline",
    "description": "Single-task baseline for BC2GM",
    "datasets": ["bc2gm"],  # ONE dataset at a time
    "model_name": "bert-base-uncased",
    "num_epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "track_tokens": True,  # CRITICAL for RQ5
    "use_wandb": False,
}
```

**Run for each combination**:
- Models: `bert-base-uncased`, `dmis-lab/biobert-v1.1`, `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract`
- Datasets: `bc2gm`, `jnlpba`, `chemprot`, `ddi`, `gad`, `hoc`, `pubmedqa`, `biosses`

**Total runs**: 3 models × 8 datasets = **24 experiments**

---

### B. Multi-Task Learning (S2) - Week 8-9

**Purpose**: Train on multiple datasets simultaneously

```python
CONFIG = {
    "experiment_type": "multi_task_shared",
    "description": "Multi-task with shared adapter",
    "datasets": ["bc2gm", "jnlpba", "chemprot"],  # Multiple datasets
    "model_name": "dmis-lab/biobert-v1.1",
    "num_epochs": 3,
    "batch_size": 8,  # Smaller due to multiple tasks
    "learning_rate": 2e-5,
    "track_tokens": True,  # Track total tokens across all tasks
    "use_wandb": True,
}
```

**Task combinations to try**:
- NER tasks: `["bc2gm", "jnlpba"]`
- RE tasks: `["chemprot", "ddi"]`
- Mixed: `["bc2gm", "chemprot", "gad"]`
- All 8: `["bc2gm", "jnlpba", "chemprot", "ddi", "gad", "hoc", "pubmedqa", "biosses"]`

---

### C. Token-Controlled Baseline (RQ5) - Week 9 ⭐ CRITICAL

**Purpose**: Fair comparison controlling for data exposure

```python
CONFIG = {
    "experiment_type": "token_controlled_baseline",
    "description": "BC2GM trained with same tokens as multi-task",
    "datasets": ["bc2gm"],  # Single task
    "model_name": "dmis-lab/biobert-v1.1",
    "num_epochs": 10,  # More epochs to match token count
    "batch_size": 16,
    "learning_rate": 2e-5,
    "track_tokens": True,
    "target_tokens": 5000000,  # SET THIS from multi-task run
}
```

**How to use**:
1. Run multi-task experiment first
2. Note total token count from results JSON
3. Set `target_tokens` to match that count
4. Run single-task with early stopping at token limit

**Example**:
- Multi-task run: 5.2M tokens across bc2gm+jnlpba
- Single-task BC2GM: Train until 5.2M tokens seen
- Compare: If multi-task still better → genuine transfer!

---

### D. Model Size Comparison (RQ3) - Week 10

**Purpose**: Find optimal model size vs performance tradeoff

```python
# Small model
CONFIG = {
    "experiment_type": "model_size_comparison",
    "model_name": "prajjwal1/bert-tiny",  # 4.4M params
    "datasets": ["bc2gm"],
    # ... rest same
}

# Medium model
CONFIG["model_name"] = "bert-base-uncased"  # 110M params

# Large model
CONFIG["model_name"] = "dmis-lab/biobert-v1.1"  # 110M params
```

---

### E. Hyperparameter Search - Week 11

**Purpose**: Find best hyperparameters for each model

```python
# Grid search over these values
learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
batch_sizes = [8, 16, 32]
warmup_ratios = [0.0, 0.1, 0.2]

for lr in learning_rates:
    for bs in batch_sizes:
        CONFIG = {
            "experiment_type": "hyperparameter_search",
            "learning_rate": lr,
            "batch_size": bs,
            "warmup_steps": int(total_steps * warmup_ratio),
            # ...
        }
        # Run experiment
```

---

## Result Organization

### After Each Experiment

Kaggle saves to `results/` directory:

```
results/
├── config_20260207_143022.json          # Experiment configuration
├── dataset_stats_20260207_143022.json   # Dataset statistics
├── results_20260207_143022.json         # Full results (USE THIS FOR PAPER)
└── summary_20260207_143022.txt          # Human-readable summary

models/
└── model_20260207_143022/
    ├── pytorch_model.bin                # Trained weights
    ├── config.json                      # Model config
    └── tokenizer files
```

### Download Results

1. Kaggle notebook → Add Output → Create Dataset
2. Include `results/` and `models/` directories
3. Download dataset after session ends
4. Analyze locally with Python/R

---

## Analyzing Results

### Load All Results (Python)

```python
import json
from pathlib import Path

results_dir = Path("results")
all_results = []

for result_file in results_dir.glob("results_*.json"):
    with open(result_file) as f:
        data = json.load(f)
        all_results.append(data)

# Create comparison table
import pandas as pd

comparison = []
for r in all_results:
    comparison.append({
        'experiment_type': r['config']['experiment_type'],
        'model': r['config']['model_name'],
        'dataset': r['config']['datasets'][0],
        'f1': r['test_results']['eval_f1'],
        'precision': r['test_results']['eval_precision'],
        'recall': r['test_results']['eval_recall'],
        'tokens': r.get('token_count', 0),
    })

df = pd.DataFrame(comparison)
print(df.sort_values('f1', ascending=False))
```

### Statistical Comparison

```python
from scipy import stats

# Compare single-task vs multi-task
single_task_f1 = df[df['experiment_type'] == 'single_task']['f1']
multi_task_f1 = df[df['experiment_type'] == 'multi_task']['f1']

t_stat, p_value = stats.ttest_ind(single_task_f1, multi_task_f1)
print(f"T-test: t={t_stat:.3f}, p={p_value:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((single_task_f1.std()**2 + multi_task_f1.std()**2) / 2)
cohens_d = (multi_task_f1.mean() - single_task_f1.mean()) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")
```

---

## Paper Tables & Figures

### Table 1: Single-Task Baselines

```python
# Generate LaTeX table
single_task_results = df[df['experiment_type'] == 'single_task']
pivot = single_task_results.pivot(index='dataset', columns='model', values='f1')
print(pivot.to_latex(float_format="%.3f"))
```

### Table 2: Multi-Task vs Token-Controlled

```python
# Compare for RQ5
comparison = df[df['experiment_type'].isin(['multi_task', 'token_controlled'])]
pivot = comparison.pivot(index='dataset', columns='experiment_type', values='f1')
pivot['improvement'] = pivot['multi_task'] - pivot['token_controlled']
print(pivot.to_latex(float_format="%.3f"))
```

### Figure 1: Token Count vs Performance

```python
import matplotlib.pyplot as plt

plt.scatter(df['tokens'], df['f1'], alpha=0.6)
plt.xlabel('Total Tokens Seen')
plt.ylabel('F1 Score')
plt.title('Token Efficiency Analysis (RQ5)')
plt.savefig('token_efficiency.pdf')
```

---

## Common Configurations

### Quick Test (5 minutes)
```python
CONFIG = {
    "datasets": ["bc2gm"],
    "max_samples_per_dataset": 100,  # Tiny subset
    "num_epochs": 1,
    "batch_size": 8,
}
```

### Full Paper Experiment (2-3 hours)
```python
CONFIG = {
    "datasets": ["bc2gm", "jnlpba"],
    "max_samples_per_dataset": None,  # All data
    "num_epochs": 3,
    "batch_size": 16,
    "use_wandb": True,  # Track everything
}
```

### Budget GPU Run (< $5)
```python
CONFIG = {
    "datasets": ["bc2gm"],
    "num_epochs": 3,
    "batch_size": 32,  # Larger batch = faster
    "save_steps": 1000,  # Less frequent checkpoints
}
```

---

## Troubleshooting

### Out of Memory (OOM)
```python
CONFIG['batch_size'] = 8  # Reduce
CONFIG['max_length'] = 256  # Reduce sequence length
```

### Training Too Slow
```python
CONFIG['num_epochs'] = 1  # Quick test first
CONFIG['max_samples_per_dataset'] = 1000  # Subset
```

### Results Not Saving
```python
# Check if results directory exists
!ls -la results/
# Manually save
import shutil
shutil.copytree('results/', '/kaggle/working/my_results/')
```

---

## Weekly Experiment Schedule

### Week 1: Single-Task Baselines
- Monday: bc2gm, jnlpba (NER)
- Tuesday: chemprot, ddi (RE)
- Wednesday: gad, hoc (Classification)
- Thursday: pubmedqa, biosses (QA/Similarity)
- Friday: Analysis + choose best model

### Week 2: Multi-Task Experiments
- Monday: NER pairs
- Tuesday: RE pairs
- Wednesday: Mixed task types
- Thursday: All 8 datasets
- Friday: Token counting + analysis

### Week 3: Token-Controlled Baselines
- Monday-Thursday: Run token-controlled for each multi-task
- Friday: Statistical comparison (RQ5 answer!)

### Week 4: Paper Writing
- Use generated results
- Create tables/figures
- Statistical tests
- Write discussion

---

## Key Metrics for Paper

From each `results_*.json`:

1. **Performance**: `test_results.eval_f1`, `eval_precision`, `eval_recall`
2. **Efficiency**: `token_count`, `train_runtime`, `train_samples_per_second`
3. **Model Size**: `model_params.trainable`, `model_params.total`
4. **Fairness**: Compare token counts between single/multi-task (RQ5)

---

## Questions?

Check:
- `CLAUDE.md` - Master orchestration file
- `MEMORY.md` - Implementation notes
- `TRAINING_EVALUATION_SUMMARY.md` - Evaluation details

**Good luck with your research! 🚀**
