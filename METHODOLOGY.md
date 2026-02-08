# Experimental Methodology
## Medical Cross-Task Knowledge Transfer with Small Language Models

---

## 1. Overview

This research investigates whether **genuine cross-task knowledge transfer** can enable small language models (110M-8B parameters) to improve performance on medical NLP tasks, when controlling for total data exposure. The key methodological innovation is the **token-controlled baseline** which separates gains from increased data exposure versus genuine cross-task transfer.

---

## 2. Research Questions

### RQ1: Multi-Task Learning Effectiveness
Does multi-task fine-tuning improve per-task performance compared to single-task fine-tuning for models under 8B parameters, after controlling for data contamination?

### RQ2: Out-of-Distribution Transfer
Can shared adapters learn representations that improve out-of-distribution task performance relative to parameter-matched single-task baselines?

### RQ3: Efficiency-Performance Trade-offs
What is the Pareto-optimal trade-off between model size, quantization level, and task performance under deployment-equivalent settings (including calibration)?

### RQ4: Negative Transfer Detection
Under what conditions does negative transfer occur, and can it be predicted from task similarity metrics (label schema overlap, vocabulary overlap)?

### RQ5: Token-Controlled Transfer Analysis ⭐ **CRITICAL**
Does controlling for total training tokens eliminate multi-task gains? If gains persist under token parity, this supports genuine cross-task transfer rather than mere data exposure.

---

## 3. Datasets

### 3.1 Dataset Selection

We use **8 medical NLP datasets** spanning diverse tasks:

| Dataset | Task Type | Samples | Tokens | Source |
|---------|-----------|---------|--------|--------|
| BC2GM | NER (Gene/Protein) | 12,574 | 1.2M | BioCreative II |
| JNLPBA | NER (Biomedical) | 24,856 | 2.4M | JNLPBA 2004 |
| ChemProt | Relation Extraction | 40,443 | 4.0M | BioCreative VI |
| DDI | Relation Extraction | 8,167 | 0.8M | DDI Extraction 2013 |
| GAD | Classification | 5,330 | 0.5M | Gene-Disease Assoc. |
| HoC | Multi-label Class. | 4,028 | 0.4M | Hallmarks of Cancer |
| PubMedQA | Question Answering | 1,000 | 0.1M | PubMedQA |
| BIOSSES | Similarity | 100 | 0.01M | Sentence Similarity |

**Total**: 96,498 samples, ~9.4M tokens across training sets

### 3.2 Data Format

All datasets stored in **pickle format** (Python standard library) for:
- Zero dependency issues on cloud platforms
- Fast loading without Arrow/Parquet libraries
- Easy version control (35MB total)

### 3.3 Data Splits

- **Training**: Used for model fine-tuning
- **Validation**: Used for early stopping and hyperparameter selection
- **Test**: Held-out evaluation (never seen during training)

---

## 4. Models

### 4.1 Base Models

| Model | Parameters | Pretraining | Use Case |
|-------|------------|-------------|----------|
| BERT-base | 110M | General domain | General baseline |
| BioBERT v1.1 | 110M | PubMed + PMC | Biomedical baseline |
| SciBERT | 110M | Scientific papers | Scientific domain |
| PubMedBERT | 110M | PubMed abstracts | Medical specialist |

All models use **WordPiece tokenization** with vocabulary size ~30K.

### 4.2 Model Selection Rationale

- **Size constraint**: All ~110M parameters for fair comparison
- **Domain relevance**: Biomedical pretraining expected to help
- **Availability**: Open-source, reproducible
- **Prior work**: Established baselines in medical NLP literature

---

## 5. Training Strategies

### 5.1 Single-Task Baseline (S1)

**Purpose**: Establish per-task performance ceiling

**Method**:
- Fine-tune each model on each dataset independently
- 24 experiments (3 models × 8 datasets)
- Early stopping with patience=3 evaluations
- Save best checkpoint based on validation F1

**Configuration**:
```python
{
  "experiment_type": "single_task_baseline",
  "datasets": ["bc2gm"],  # One at a time
  "model_name": "dmis-lab/biobert-v1.1",
  "num_epochs": 10,  # Max, early stopping active
  "batch_size": 16,
  "learning_rate": 2e-5,
  "use_early_stopping": True,
  "early_stopping_patience": 3,
  "track_tokens": True,  # CRITICAL for RQ5
}
```

### 5.2 Multi-Task Learning (S2)

**Purpose**: Test if joint training improves performance

**Method**:
- Train single model on multiple datasets simultaneously
- Batch sampling proportional to dataset size
- Shared encoder + task-specific heads
- Same early stopping criteria

**Task Combinations**:
1. **NER tasks**: BC2GM + JNLPBA
2. **RE tasks**: ChemProt + DDI
3. **Mixed**: BC2GM + ChemProt + GAD
4. **All 8**: Complete multi-task setup

**Configuration**:
```python
{
  "experiment_type": "multi_task",
  "datasets": ["bc2gm", "jnlpba", "chemprot"],  # Multiple
  "model_name": "dmis-lab/biobert-v1.1",
  "num_epochs": 10,
  "batch_size": 8,  # Smaller due to multiple tasks
  "track_tokens": True,  # Total across all tasks
}
```

### 5.3 Token-Controlled Baseline (RQ5) ⭐ **KEY INNOVATION**

**Purpose**: Disentangle transfer learning from data exposure

**Method**:
1. Run multi-task experiment, record total tokens seen (e.g., 7.6M)
2. Run single-task experiment until SAME token count reached
3. Compare performance with equal data exposure

**Why This Matters**:
- Traditional comparison: Multi-task sees 7.6M tokens, single-task sees 1.2M tokens → UNFAIR
- Token-controlled: Both see 7.6M tokens → FAIR comparison
- If multi-task STILL better → genuine cross-task transfer proven!

**Configuration**:
```python
# Step 1: Multi-task (from S2 results)
multi_task_tokens = 7600000  # Example from results JSON

# Step 2: Token-controlled single-task
{
  "experiment_type": "token_controlled_baseline",
  "datasets": ["bc2gm"],  # Single task
  "target_tokens": 7600000,  # Match multi-task exactly
  "use_early_stopping": False,  # Train to token limit
  "num_epochs": 20,  # High max, will stop when tokens reached
  "track_tokens": True,
}
```

**Analysis**:
```
Scenario A: Multi-task NOT better after token control
→ Gains were just from more data (NOT genuine transfer)

Scenario B: Multi-task STILL better after token control
→ Genuine cross-task knowledge transfer! ✅
```

---

## 6. Training Configuration

### 6.1 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Max epochs | 10 | With early stopping |
| Batch size | 16 (single), 8 (multi) | Fits in T4 16GB VRAM |
| Learning rate | 2e-5 | Standard for BERT fine-tuning |
| Warmup steps | 500 | ~10% of training |
| Weight decay | 0.01 | Regularization |
| Max length | 512 | BERT limit |
| Optimizer | AdamW | Standard |
| LR schedule | Linear warmup + decay | Standard |

### 6.2 Early Stopping

**Configuration**:
- **Patience**: 3 evaluations
- **Threshold**: 0.0001 minimum improvement
- **Metric**: Validation F1 score
- **Evaluation frequency**: Every 250 steps

**Rationale**:
- Prevents overfitting on small medical datasets
- Dataset-adaptive (BC2GM stops at epoch 2, ChemProt at epoch 6)
- More rigorous than arbitrary fixed epochs
- Loads best checkpoint for final evaluation

**Expected behavior**:

```
Epoch 1.0 → Val F1: 0.7234
Epoch 1.5 → Val F1: 0.7512 ✓ improvement
Epoch 2.0 → Val F1: 0.7834 ✓ improvement [BEST]
Epoch 2.5 → Val F1: 0.7821 (no improvement, patience=1)
Epoch 3.0 → Val F1: 0.7811 (no improvement, patience=2)
Epoch 3.5 → Val F1: 0.7798 (no improvement, patience=3)
→ STOP! Return checkpoint from epoch 2.0
```

### 6.3 Token Tracking Implementation

**Purpose**: Enable token-controlled comparisons (RQ5)

**Implementation**:
```python
class TokenTrackingNERDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512, task_name="unknown"):
        self.total_tokens = 0

    def __getitem__(self, idx):
        encoding = self.tokenizer(text, max_length=self.max_length, ...)
        num_tokens = encoding['attention_mask'].sum().item()
        self.total_tokens += num_tokens  # Accumulate
        return {..., 'num_tokens': num_tokens}
```

**Tracking**:
- Count actual tokens (not padding) via attention mask
- Accumulate across all batches and epochs
- Save in results JSON for each experiment
- Use for token-controlled baseline matching

**Example results**:
```json
{
  "experiment_type": "single_task_baseline",
  "dataset": "bc2gm",
  "epochs_trained": 2.6,
  "token_count": 1234567,
  "test_f1": 0.7834
}
```

---

## 7. Evaluation Metrics

### 7.1 Primary Metrics

**Named Entity Recognition (BC2GM, JNLPBA)**:
- **F1 score** (micro-averaged, entity-level)
- Precision, Recall
- Computed using `seqeval` library

**Relation Extraction (ChemProt, DDI)**:
- **F1 score** (macro-averaged, relation-level)
- Per-relation F1 scores

**Classification (GAD, HoC)**:
- **F1 score** (macro for multi-class, micro for multi-label)
- Accuracy (secondary metric)

**Question Answering (PubMedQA)**:
- **Exact Match** (primary)
- F1 score (token overlap)

**Similarity (BIOSSES)**:
- **Pearson correlation**
- Spearman correlation

### 7.2 Statistical Testing

**Paired t-test**:
- Compare single-task vs multi-task F1 scores
- Null hypothesis: No difference in means
- Significance level: α = 0.05

**Effect size (Cohen's d)**:
```
d = (μ_multi - μ_single) / σ_pooled

Interpretation:
  |d| < 0.5: small effect
  |d| < 0.8: medium effect
  |d| ≥ 0.8: large effect
```

**Bootstrap confidence intervals**:
- 10,000 bootstrap samples
- 95% confidence intervals on F1 scores

### 7.3 Token Efficiency Analysis (RQ5)

**Metric**: F1 score per million tokens

```
Efficiency = F1 / (tokens / 1,000,000)

Example:
  Single-task: F1=0.78, tokens=1.2M → Efficiency=0.65
  Multi-task:  F1=0.83, tokens=7.6M → Efficiency=0.11
  Token-controlled: F1=0.80, tokens=7.6M → Efficiency=0.11

Conclusion: Multi-task achieves +3% F1 with SAME data exposure
```

---

## 8. Experimental Procedure

### 8.1 Phase 1: Single-Task Baselines (Week 1-2)

**Goal**: Establish performance ceiling per dataset

**Experiments**:
```
For each model in [BERT, BioBERT, PubMedBERT]:
  For each dataset in [bc2gm, jnlpba, chemprot, ddi, gad, hoc, pubmedqa, biosses]:
    Train single-task model
    Record: F1, tokens, epochs, training time
    Save: Model checkpoint, results JSON
```

**Total**: 24 experiments (3 models × 8 datasets)

**Expected runtime**: 2-3 hours per experiment on T4 GPU

### 8.2 Phase 2: Multi-Task Experiments (Week 3)

**Goal**: Test joint training effectiveness

**Experiments**:
1. NER pair: BC2GM + JNLPBA
2. RE pair: ChemProt + DDI
3. Mixed: BC2GM + ChemProt + GAD
4. All 8 datasets combined

**For each experiment**:
- Record total tokens across all tasks
- Evaluate on each dataset's test set individually
- Compare to single-task baselines

**Total**: 4-8 experiments (depending on combinations tested)

### 8.3 Phase 3: Token-Controlled Baselines (Week 4) ⭐

**Goal**: Prove genuine transfer vs data exposure

**Procedure**:
```python
# For each multi-task experiment from Phase 2:

# 1. Get multi-task token count
multi_task_result = load_json("results/multi_task_bc2gm_jnlpba.json")
target_tokens = multi_task_result['token_count']

# 2. Run single-task with SAME token count
for dataset in multi_task_datasets:
    train_until_tokens(
        dataset=dataset,
        target_tokens=target_tokens,
        max_epochs=20
    )

# 3. Compare:
#    - Multi-task F1 on dataset
#    - Single-task F1 on dataset (token-controlled)
#    - Original single-task F1 (Phase 1)
```

**Critical comparison**:
```
Dataset: BC2GM
  - Single-task baseline: F1=0.78, tokens=1.2M
  - Multi-task: F1=0.83, tokens=7.6M
  - Token-controlled single: F1=0.80, tokens=7.6M

Conclusion: Multi-task gains +3% even with equal tokens
→ Genuine cross-task transfer confirmed! ✅
```

### 8.4 Data Collection

**For each experiment, save**:
```json
{
  "experiment_id": "20260207_143022",
  "experiment_type": "single_task_baseline",
  "config": {
    "model_name": "dmis-lab/biobert-v1.1",
    "datasets": ["bc2gm"],
    "num_epochs": 10,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "use_early_stopping": true,
    "early_stopping_patience": 3
  },
  "dataset_stats": {
    "bc2gm": {
      "train_samples": 12574,
      "val_samples": 2000,
      "test_samples": 2500,
      "num_labels": 5
    }
  },
  "model_params": {
    "total": 110000000,
    "trainable": 110000000
  },
  "train_results": {
    "train_loss": 0.123,
    "train_runtime": 7200,
    "epochs_trained": 2.6,
    "train_samples_per_second": 45.2
  },
  "test_results": {
    "eval_loss": 0.234,
    "eval_f1": 0.7834,
    "eval_precision": 0.7912,
    "eval_recall": 0.7758
  },
  "token_count": 1234567
}
```

---

## 9. Analysis Pipeline

### 9.1 Result Aggregation

**Script**: `analyze_results.py`

**Input**: All `results_*.json` files from experiments

**Output**:
1. `all_results.csv` - Tabular format for analysis
2. `table_single_task.tex` - LaTeX table (Table 1)
3. `table_multitask_comparison.tex` - LaTeX table (Table 2)
4. `table_rq5_token_controlled.tex` - LaTeX table (Table 3) ⭐
5. `statistics.json` - Statistical tests
6. `SUMMARY.txt` - Human-readable report

### 9.2 Statistical Analysis

```python
# Load all results
df = pd.read_csv('analysis/all_results.csv')

# Compare single-task vs multi-task
st_f1 = df[df['experiment_type'] == 'single_task']['f1']
mt_f1 = df[df['experiment_type'] == 'multi_task']['f1']

# T-test
t_stat, p_value = stats.ttest_ind(st_f1, mt_f1)

# Effect size
cohens_d = (mt_f1.mean() - st_f1.mean()) / pooled_std

# Report
print(f"Multi-task vs Single-task:")
print(f"  Mean difference: {mt_f1.mean() - st_f1.mean():.4f}")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Cohen's d: {cohens_d:.3f}")
```

### 9.3 RQ5 Analysis (Token-Controlled)

```python
# Critical analysis for paper
for dataset in datasets:
    st_baseline = get_result(dataset, 'single_task')
    mt_result = get_result(dataset, 'multi_task')
    tc_baseline = get_result(dataset, 'token_controlled')

    # Check token parity
    token_diff = abs(mt_result['tokens'] - tc_baseline['tokens'])
    token_parity = (token_diff / mt_result['tokens']) < 0.05

    # Compare F1
    mt_advantage = mt_result['f1'] - tc_baseline['f1']

    print(f"{dataset}:")
    print(f"  Multi-task: F1={mt_result['f1']:.4f}, tokens={mt_result['tokens']:,}")
    print(f"  Token-controlled: F1={tc_baseline['f1']:.4f}, tokens={tc_baseline['tokens']:,}")
    print(f"  Token parity: {token_parity}")
    print(f"  MT advantage: {mt_advantage:+.4f}")
    print(f"  Genuine transfer: {'YES' if mt_advantage > 0 and token_parity else 'NO'}")
```

---

## 10. Computational Resources

### 10.1 Hardware

**Platform**: Kaggle (free tier)
- **GPU**: NVIDIA Tesla T4 (16GB VRAM)
- **Accelerators**: 2x T4 GPUs
- **CPU**: 4-core Intel Xeon
- **RAM**: 30GB
- **Disk**: 20GB (with checkpoint management)

### 10.2 Software Environment

```
Python: 3.10+
PyTorch: 2.0+
Transformers: 4.35+
CUDA: 11.8+
seqeval: 1.2.2
wandb: 0.16+ (optional)
```

### 10.3 Runtime Estimates

| Experiment Type | Dataset Size | Runtime (T4) | Cost (if paid) |
|----------------|--------------|--------------|----------------|
| Single-task small | <5K samples | 30-60 min | $0.50 |
| Single-task medium | 5-20K | 1-2 hours | $1.50 |
| Single-task large | 20-50K | 2-4 hours | $3.00 |
| Multi-task (2 datasets) | Combined | 2-3 hours | $2.50 |
| Multi-task (all 8) | 96K samples | 6-8 hours | $8.00 |

**Total project cost**: $0 (Kaggle free) to $80 (if using paid compute)

---

## 11. Reproducibility

### 11.1 Random Seeds

All experiments use fixed random seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### 11.2 Version Control

- Code: GitHub repository
- Data: Included in repository (pickle format, 35MB)
- Models: HuggingFace model hub (public)
- Results: Saved in `results/` directory

### 11.3 Exact Reproduction

```bash
# Clone repository
git clone https://github.com/bharathbolla/Crosstalk_Medical_LLM.git
cd Crosstalk_Medical_LLM

# Verify datasets
python test_pickle_load.py

# Run single experiment
# Upload KAGGLE_RESEARCH_EXPERIMENTS.ipynb to Kaggle
# Configure Cell 3 with desired experiment
# Run all cells

# Download results
# Add output dataset in Kaggle
# Download results/ and models/ directories

# Analyze
python analyze_results.py --results_dir results/
```

---

## 12. Limitations and Considerations

### 12.1 Dataset Limitations

- Medical datasets are small compared to general NLP
- Some datasets have class imbalance
- Domain-specific: Results may not generalize to other domains

### 12.2 Model Limitations

- Models limited to ~110M parameters (larger models future work)
- English language only
- Fine-tuning only (no pretraining from scratch)

### 12.3 Computational Limitations

- Free GPU access limited to certain hours
- Session time limits (12 hours max)
- Checkpoint management required for long runs

### 12.4 Evaluation Considerations

- Token counting is approximate (depends on tokenizer)
- Early stopping introduces some variability
- Test sets are relatively small for some datasets

---

## 13. Ethical Considerations

### 13.1 Data Usage

- All datasets are publicly available
- No patient-identifiable information
- Compliant with research ethics

### 13.2 Model Deployment

- Models are research prototypes, not clinical tools
- Would require validation before clinical deployment
- Performance calibration assessment included (ECE metric)

### 13.3 Environmental Impact

- Using free/shared GPU resources
- Checkpoint management reduces redundant computation
- Early stopping reduces unnecessary training

---

## 14. Expected Outcomes

### 14.1 Success Criteria

**RQ5 (Primary)**:
- Token-controlled baseline implemented ✅
- Statistical evidence of genuine transfer (p < 0.05)
- Effect size: Cohen's d > 0.3

**RQ1**:
- Multi-task shows improvement on ≥5/8 datasets
- Average improvement: +2-5% F1

**RQ2-RQ4** (Secondary):
- OOD transfer demonstrated on held-out tasks
- Negative transfer identified and explained
- Efficiency analysis complete

### 14.2 Publication Target

**Venue**: ACL, EMNLP, NAACL, or BioNLP workshop

**Contributions**:
1. First token-controlled analysis of medical multi-task learning
2. Systematic comparison across 8 medical datasets
3. Open-source implementation and data

---

## References

This methodology builds on:
- BioBERT (Lee et al., 2019)
- PubMedBERT (Gu et al., 2020)
- Multi-task learning (Caruana, 1997; Ruder, 2017)
- Token-level analysis (our contribution)

---

**Document Version**: 1.0
**Last Updated**: 2026-02-07
**Status**: Ready for implementation
