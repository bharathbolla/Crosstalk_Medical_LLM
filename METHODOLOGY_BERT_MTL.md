# Experimental Methodology
## Comparative Analysis of Multi-Task Learning Across BERT Model Variants in Medical NLP

**Research Focus**: Systematic comparison of 7 BERT-variant models (general, biomedical, clinical) on 8 medical NLP tasks with token-controlled analysis

**Key Contributions**:
1. First token-controlled baseline in medical multi-task learning
2. Comprehensive comparison of BERT vs RoBERTa architectures for medical NLP
3. Analysis of pretraining corpus effects (general vs biomedical vs clinical)

**Publication Target**: BioNLP Workshop, EMNLP Findings, Journal of Biomedical Informatics

---

## 1. Research Overview

### 1.1 Motivation

Multi-task learning (MTL) has shown promise in medical NLP, with papers reporting 3-8% F1 improvements over single-task baselines. However, **these gains conflate two distinct effects**:

1. **Data exposure effect**: Multi-task models see more total training data
2. **Transfer learning effect**: Cross-task knowledge sharing

**Critical gap**: No prior work controls for data exposure when evaluating medical MTL.

**Our contribution**: Token-controlled baseline that separates these effects.

### 1.2 Research Questions

**RQ1: Model Comparison**
How do different BERT variants (general, biomedical, clinical) perform across 8 diverse medical NLP tasks?

**RQ2: Architecture Analysis**
Does RoBERTa architecture outperform BERT for medical NLP, and is the difference consistent across tasks?

**RQ3: Pretraining Corpus Effect**
What is the relative importance of pretraining corpus (general → biomedical → clinical) for task performance?

**RQ4: Multi-Task Effectiveness**
Does multi-task learning improve performance compared to single-task baselines across all model variants?

**RQ5: Token-Controlled Analysis** ⭐ **KEY NOVELTY**
Do multi-task gains persist when controlling for total training tokens (data exposure)?

**RQ6: Model-Specific Transfer**
Do different models benefit differently from multi-task learning (e.g., does general BERT benefit more than specialized models)?

**RQ7: Task Combination Analysis**
Which task combinations benefit from joint training, and which exhibit negative transfer?

**RQ8: Practical Guidelines**
What are evidence-based guidelines for model selection and training strategy (single vs multi-task)?

---

## 2. Datasets

### 2.1 Task Selection Rationale

We select **8 medical NLP datasets** spanning 5 task types to ensure:
- **Diversity**: Different linguistic phenomena (entities, relations, semantics)
- **Size variation**: 100-40K samples (test generalization)
- **Clinical relevance**: All tasks used in real medical applications
- **Public availability**: Reproducible without IRB approval

### 2.2 Dataset Details

| Dataset | Task Type | Domain | Train Size | Tokens | Source |
|---------|-----------|--------|------------|--------|--------|
| **BC2GM** | NER (Gene/Protein) | Biomedical | 12,574 | 1.2M | BioCreative II |
| **JNLPBA** | NER (Bio-entities) | Biomedical | 24,856 | 2.4M | JNLPBA 2004 |
| **ChemProt** | Relation Extraction | Biomedical | 40,443 | 4.0M | BioCreative VI |
| **DDI** | Relation Extraction | Clinical | 8,167 | 0.8M | DDI Corpus |
| **GAD** | Binary Classification | Genetics | 5,330 | 0.5M | Gene-Disease |
| **HoC** | Multi-label Class. | Oncology | 4,028 | 0.4M | Hallmarks Cancer |
| **PubMedQA** | Question Answering | Biomedical | 1,000 | 0.1M | PubMedQA |
| **BIOSSES** | Semantic Similarity | Biomedical | 100 | 0.01M | BIOSSES |

**Total**: 96,498 training samples, ~9.4M tokens

### 2.3 Task Type Distribution

```
NER (Named Entity Recognition):
  - BC2GM: 5 entity types (Gene, Protein, etc.)
  - JNLPBA: 5 bio-entity types (DNA, RNA, Cell Line, etc.)

RE (Relation Extraction):
  - ChemProt: 13 chemical-protein interaction types
  - DDI: 4 drug-drug interaction types

Classification:
  - GAD: Binary (gene-disease association: yes/no)
  - HoC: Multi-label (10 cancer hallmarks)

QA (Question Answering):
  - PubMedQA: Yes/No/Maybe answers to biomedical questions

Similarity:
  - BIOSSES: Sentence pair similarity scores (0-4)
```

### 2.4 Data Format

**Storage**: Pickle format (Python standard library)
- No external dependencies (datasets library, pyarrow)
- Fast loading (35MB total, loads in <5 seconds)
- Platform-independent (works on Kaggle, Colab, local)

**Structure**:
```python
{
  'train': [
    {'id': '...', 'tokens': [...], 'ner_tags': [...], ...},
    ...
  ],
  'validation': [...],
  'test': [...]
}
```

### 2.5 Data Splits

- **Training**: Model fine-tuning with early stopping
- **Validation**: Hyperparameter selection, early stopping criterion
- **Test**: Final evaluation (reported in paper)

**Split sizes**:
- Train: 70-80% (dataset dependent)
- Validation: 10-15%
- Test: 10-15%

---

## 3. Models

### 3.1 Base Model Selection

We evaluate **7 BERT-variant models** spanning general, biomedical, and clinical domains:

| Model | Parameters | Architecture | Pretraining Corpus | Domain | HF Model ID |
|-------|------------|--------------|-------------------|--------|-------------|
| **BERT-base** | 110M | BERT | BooksCorpus + Wikipedia | General | `bert-base-uncased` |
| **RoBERTa-base** | 125M | RoBERTa | 160GB text (general) | General | `roberta-base` |
| **BioBERT v1.1** | 110M | BERT | PubMed + PMC (18B words) | Biomedical | `dmis-lab/biobert-v1.1` |
| **PubMedBERT** | 110M | BERT | PubMed abstracts (21B words) | Biomedical | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` |
| **Clinical-BERT** | 110M | BERT | MIMIC-III clinical notes | Clinical | `emilyalsentzer/Bio_ClinicalBERT` |
| **BlueBERT** | 110M | BERT | PubMed + MIMIC-III | Bio+Clinical | `bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12` |
| **BioMed-RoBERTa** | 125M | RoBERTa | PubMed abstracts | Biomedical | `allenai/biomed_roberta_base` |

### 3.2 Model Selection Rationale

**Why 7 diverse models?**
- **General baselines** (BERT, RoBERTa): Test if medical pretraining matters
- **Biomedical specialists** (BioBERT, PubMedBERT, BioMed-RoBERTa): Compare domain-specific pretraining strategies
- **Clinical specialist** (Clinical-BERT): Test clinical notes vs research literature pretraining
- **Hybrid** (BlueBERT): Combines biomedical + clinical data

**Why ~110-125M parameters?**
- Standard size for BERT-family models
- Fits on single GPU (T4 16GB) with batch size 32
- Directly comparable to extensive prior work
- Practical for deployment in clinical settings

**Architecture diversity**:
- **BERT** (5 models): Bidirectional encoder, masked language modeling
- **RoBERTa** (2 models): Improved BERT training (more data, dynamic masking, no NSP)
- Both proven effective for medical NLP

**Pretraining corpus diversity**:
```
General domain:
  - BERT-base: General text (16GB)
  - RoBERTa-base: General text (160GB) - 10x more data!

Biomedical research:
  - BioBERT: PubMed + PMC (continued from BERT)
  - PubMedBERT: PubMed (from-scratch training)
  - BioMed-RoBERTa: PubMed (RoBERTa architecture)

Clinical notes:
  - Clinical-BERT: MIMIC-III (real clinical text)

Hybrid:
  - BlueBERT: PubMed + MIMIC-III (best of both)
```

**Expected performance ranking** (hypothesis):
1. **BlueBERT**: Best overall (biomedical + clinical pretraining)
2. **Clinical-BERT**: Best on clinical tasks (DDI, clinical NER)
3. **PubMedBERT**: Best on biomedical research tasks (BC2GM, JNLPBA)
4. **BioMed-RoBERTa**: Strong biomedical performance (RoBERTa advantage)
5. **BioBERT**: Solid biomedical baseline
6. **RoBERTa-base**: Better than BERT (more robust training)
7. **BERT-base**: Weakest (no domain adaptation)

**Why NOT larger models?**
- Llama-3-8B, GPT-3.5: Require multiple GPUs or extensive quantization
- Focus on practically deployable models
- BERT family has established baselines for fair comparison
- All models can run on single T4 GPU

### 3.3 Model Architecture

**Task-Specific Heads** (added on top of BERT encoder):

```python
# NER tasks (BC2GM, JNLPBA)
NER_head = Linear(768 → num_labels)  # Token classification

# RE tasks (ChemProt, DDI)
RE_head = Bilinear(768 × 768 → num_relations)  # Entity pair classification

# Classification (GAD, HoC)
CLS_head = Linear(768 → num_classes)  # [CLS] token classification

# QA (PubMedQA)
QA_head = Linear(768 → 3)  # Yes/No/Maybe

# Similarity (BIOSSES)
SIM_head = Linear(768 × 2 → 1)  # Sentence pair similarity
```

**Multi-Task Architecture**:
- **Shared**: BERT encoder (all 12 transformer layers)
- **Private**: Task-specific classification heads
- **Training**: Batch sampling proportional to dataset size

---

## 4. Experimental Design

### 4.1 Training Strategies

#### Strategy 1: Single-Task Baseline (S1)

**Purpose**: Establish per-task performance ceiling

**Method**:
```python
For each model in [BERT, RoBERTa, BioBERT, PubMedBERT, ClinicalBERT, BlueBERT, BioMed-RoBERTa]:
    For each task in [BC2GM, JNLPBA, ..., BIOSSES]:
        Fine-tune model on task independently
        Save best checkpoint (validation F1)
        Record: F1, tokens, epochs, time
```

**Total experiments**: 7 models × 8 tasks = **56 experiments**

**Configuration**:
```python
{
  "experiment_type": "single_task_baseline",
  "model": "dmis-lab/biobert-v1.1",
  "dataset": "bc2gm",
  "batch_size": 32,
  "learning_rate": 2e-5,
  "num_epochs": 10,
  "early_stopping_patience": 3,
  "track_tokens": True  # Critical for RQ2!
}
```

#### Strategy 2: Multi-Task Learning (S2)

**Purpose**: Test joint training effectiveness

**Task Combinations**:
1. **NER pair**: BC2GM + JNLPBA (similar tasks)
2. **RE pair**: ChemProt + DDI (similar tasks)
3. **Mixed**: BC2GM + ChemProt + GAD (diverse tasks)
4. **All 8**: Complete multi-task setup

**Method**:
```python
For each task_combination in [NER_pair, RE_pair, Mixed, All_8]:
    For each model in [BERT, RoBERTa, BioBERT, PubMedBERT, ClinicalBERT, BlueBERT, BioMed-RoBERTa]:
        Train shared encoder + task-specific heads
        Batch sampling: proportional to dataset size
        Evaluate on each task's test set independently
        Record total tokens across all tasks
```

**Total experiments**: 4 combinations × 7 models = **28 experiments**

**Prioritization** (if time/compute limited):
- **Must do**: BlueBERT, PubMedBERT, Clinical-BERT (top 3 expected performers)
- **Should do**: BioMed-RoBERTa, BioBERT (strong biomedical models)
- **Nice to have**: BERT-base, RoBERTa-base (general baselines)

**Configuration**:
```python
{
  "experiment_type": "multi_task",
  "model": "dmis-lab/biobert-v1.1",
  "datasets": ["bc2gm", "jnlpba"],
  "batch_size": 16,  # Smaller due to multiple heads
  "learning_rate": 2e-5,
  "num_epochs": 10,
  "early_stopping_patience": 3,
  "track_tokens": True  # Total across all tasks
}
```

#### Strategy 3: Token-Controlled Baseline (S3) ⭐ **KEY NOVELTY**

**Purpose**: Separate data exposure from genuine transfer

**Hypothesis**:
- H0: Multi-task gains are due to seeing more data (data exposure)
- H1: Multi-task gains persist with equal data exposure (genuine transfer)

**Method**:
```python
# Step 1: Run multi-task (S2) and record total tokens
multi_task_result = train_multi_task(tasks=["bc2gm", "jnlpba"])
target_tokens = multi_task_result.total_tokens  # e.g., 7.6M

# Step 2: Train single-task until SAME token count
for task in ["bc2gm", "jnlpba"]:
    train_until_tokens(
        task=task,
        target_tokens=target_tokens,  # Match multi-task!
        max_epochs=20  # High limit, stops when tokens reached
    )

# Step 3: Compare
compare(
    single_task_baseline,     # 1.2M tokens
    multi_task,               # 7.6M tokens
    token_controlled_single   # 7.6M tokens (matched!)
)
```

**Critical comparison**:
```
Task: BC2GM
  Single-task baseline:  F1=0.78, tokens=1.2M
  Multi-task:           F1=0.83, tokens=7.6M (+5% F1)
  Token-controlled:     F1=0.80, tokens=7.6M

Analysis:
  Traditional comparison: +5% F1 (but unfair, 6.3x more data!)
  Token-controlled: +3% F1 (fair comparison, same data)

Conclusion:
  - 2% gain from data exposure (not interesting)
  - 3% gain from genuine transfer (THIS IS THE CONTRIBUTION!)
```

**Total experiments**: 4 combinations × 8 tasks × 7 models = **224 experiments**

**Realistic subset** (focus on best models):
- 4 combinations × 8 tasks × 3 models (BlueBERT, PubMedBERT, Clinical-BERT) = **96 experiments**
- This is the most important analysis for RQ2!

**Prioritization strategy**:
```python
# Priority 1: Best performing models from S1
top_3_models = identify_best_models_from_s1()  # e.g., BlueBERT, PubMedBERT, Clinical-BERT

# Priority 2: Token-controlled for top models only
for model in top_3_models:
    for combination in [NER_pair, RE_pair, Mixed, All_8]:
        run_token_controlled_baseline(model, combination)

# Priority 3: If time permits, extend to all 7 models
```

### 4.2 Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Batch size** | 32 (single), 16 (multi) | Optimal for T4 16GB VRAM |
| **Learning rate** | 2e-5 | Standard BERT fine-tuning (Devlin et al., 2019) |
| **Warmup ratio** | 0.1 | 10% of training steps |
| **Weight decay** | 0.01 | L2 regularization |
| **Max sequence length** | 512 | BERT limit |
| **Optimizer** | AdamW | Standard for transformers |
| **LR schedule** | Linear decay | After warmup |
| **Precision** | FP16 | 2x faster, same performance |
| **Max epochs** | 10 | With early stopping |
| **Early stopping patience** | 3 | Evaluations without improvement |
| **Early stopping threshold** | 0.0001 | Minimum meaningful improvement |
| **Evaluation frequency** | Every 250 steps | For early stopping |

**Hyperparameter search** (for best model only):
```python
learning_rates = [1e-5, 2e-5, 3e-5, 5e-5]
batch_sizes = [16, 24, 32]
warmup_ratios = [0.0, 0.1, 0.2]

# Grid search on validation set
# Report best configuration in appendix
```

### 4.3 Early Stopping

**Rationale**: Medical datasets are small (100-40K samples), prone to overfitting

**Configuration**:
- **Metric**: Validation F1 score
- **Patience**: 3 evaluations without improvement
- **Threshold**: Minimum 0.0001 improvement to count
- **Evaluation**: Every 250 training steps

**Behavior**:
```
Step 250:  Val F1 = 0.7234 [Best]
Step 500:  Val F1 = 0.7512 [Best] ✓
Step 750:  Val F1 = 0.7834 [Best] ✓
Step 1000: Val F1 = 0.7821 (worse, patience=1)
Step 1250: Val F1 = 0.7798 (worse, patience=2)
Step 1500: Val F1 = 0.7811 (worse, patience=3)
→ STOP! Load checkpoint from step 750
```

**Advantages**:
- Prevents overfitting (especially on small datasets like BIOSSES)
- Dataset-adaptive (BC2GM stops at epoch 2, ChemProt at epoch 6)
- More rigorous than fixed epochs
- Standard practice in NLP (Lewis et al., 2020)

### 4.4 Token Tracking Implementation

**Purpose**: Enable token-controlled comparisons (RQ2)

**Implementation**:
```python
class TokenTrackingDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.total_tokens = 0

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            max_length=512,
            truncation=True,
            padding='max_length'
        )

        # Count non-padding tokens
        num_tokens = encoding['attention_mask'].sum()
        self.total_tokens += num_tokens

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': item['labels'],
            'num_tokens': num_tokens
        }
```

**Tracking across training**:
```python
# After each epoch
epoch_tokens = train_dataset.total_tokens
cumulative_tokens += epoch_tokens

# Save in results
results = {
    'epochs_trained': trainer.state.epoch,
    'total_tokens': cumulative_tokens,
    'test_f1': test_results['f1'],
    ...
}
```

**Token-controlled stopping**:
```python
def should_stop_training(current_tokens, target_tokens):
    if target_tokens is None:
        return False  # Normal early stopping
    return current_tokens >= target_tokens
```

---

## 5. Evaluation

### 5.1 Primary Metrics (By Task Type)

**Named Entity Recognition** (BC2GM, JNLPBA):
```python
from seqeval.metrics import f1_score, precision_score, recall_score

# Entity-level F1 (strict matching)
f1 = f1_score(true_labels, predictions, mode='strict')
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
```

**Relation Extraction** (ChemProt, DDI):
```python
# Macro F1 (average across relation types)
f1_macro = f1_score(y_true, y_pred, average='macro')

# Per-relation F1 (for analysis)
f1_per_relation = f1_score(y_true, y_pred, average=None)
```

**Classification** (GAD, HoC):
```python
# GAD (binary): F1 for positive class
f1_positive = f1_score(y_true, y_pred, pos_label=1)

# HoC (multi-label): Micro F1 (treats each label independently)
f1_micro = f1_score(y_true, y_pred, average='micro')
```

**Question Answering** (PubMedQA):
```python
# Exact match (primary metric)
exact_match = (predictions == ground_truth).mean()

# Accuracy (same for this task)
accuracy = accuracy_score(ground_truth, predictions)
```

**Similarity** (BIOSSES):
```python
from scipy.stats import pearsonr, spearmanr

# Pearson correlation (primary)
pearson_r, p_value = pearsonr(predictions, ground_truth)

# Spearman correlation (secondary)
spearman_rho, p_value = spearmanr(predictions, ground_truth)
```

### 5.2 Statistical Testing

**Paired t-test** (compare single-task vs multi-task):
```python
from scipy.stats import ttest_rel

# Null hypothesis: No difference in means
# Alternative: Multi-task > Single-task
t_stat, p_value = ttest_rel(multi_task_f1s, single_task_f1s)

# Report: t(df=7) = X.XX, p < 0.05
```

**Effect size** (Cohen's d):
```python
import numpy as np

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

d = cohens_d(multi_task_f1s, single_task_f1s)

# Interpretation:
# |d| < 0.5: small effect
# 0.5 ≤ |d| < 0.8: medium effect
# |d| ≥ 0.8: large effect
```

**Bootstrap confidence intervals**:
```python
from scipy.stats import bootstrap

def mean_f1(data, axis):
    return np.mean(data, axis=axis)

# 95% CI via bootstrap
rng = np.random.default_rng(seed=42)
res = bootstrap(
    (f1_scores,),
    mean_f1,
    n_resamples=10000,
    confidence_level=0.95,
    random_state=rng
)

ci_lower, ci_upper = res.confidence_interval
```

### 5.3 Token-Controlled Analysis (RQ2)

**Critical comparison table**:

| Dataset | ST Baseline | Multi-Task | Token-Controlled | Genuine Transfer? |
|---------|-------------|------------|------------------|-------------------|
|         | F1 / Tokens | F1 / Tokens | F1 / Tokens     | MT > TC?          |
| BC2GM   | 0.78 / 1.2M | 0.83 / 7.6M | 0.80 / 7.6M     | ✓ (+3%)           |
| JNLPBA  | 0.75 / 2.4M | 0.79 / 7.6M | 0.77 / 7.6M     | ✓ (+2%)           |
| ChemProt| 0.71 / 4.0M | 0.74 / 7.6M | 0.73 / 7.6M     | ✓ (+1%)           |

**Analysis**:
```python
for task in tasks:
    st = single_task_results[task]
    mt = multi_task_results[task]
    tc = token_controlled_results[task]

    # Verify token parity
    token_diff_pct = abs(mt['tokens'] - tc['tokens']) / mt['tokens']
    assert token_diff_pct < 0.05, "Token parity violated!"

    # Compute gains
    traditional_gain = mt['f1'] - st['f1']
    controlled_gain = mt['f1'] - tc['f1']

    # Report
    print(f"{task}:")
    print(f"  Traditional gain: +{traditional_gain:.3f}")
    print(f"  Controlled gain: +{controlled_gain:.3f}")
    print(f"  Data exposure effect: {traditional_gain - controlled_gain:.3f}")
```

**Statistical test** (does MT beat TC?):
```python
# Paired t-test: MT vs TC (with equal tokens)
t_stat, p_value = ttest_rel(mt_f1s, tc_f1s)

if p_value < 0.05 and mean(mt_f1s) > mean(tc_f1s):
    conclusion = "Genuine cross-task transfer confirmed!"
else:
    conclusion = "No evidence of genuine transfer (gains due to data exposure)"
```

### 5.4 Negative Transfer Analysis (RQ3)

**Definition**: Negative transfer occurs when multi-task < single-task

**Detection**:
```python
for task in tasks:
    st_f1 = single_task_results[task]['f1']
    mt_f1 = multi_task_results[task]['f1']

    if mt_f1 < st_f1:
        print(f"⚠️ Negative transfer on {task}: {mt_f1 - st_f1:.3f}")
```

**Task similarity metrics**:
```python
def label_overlap(task1, task2):
    """Jaccard similarity of label sets"""
    labels1 = set(task1.label_names)
    labels2 = set(task2.label_names)
    return len(labels1 & labels2) / len(labels1 | labels2)

def vocabulary_overlap(task1, task2):
    """Jaccard similarity of vocabularies"""
    vocab1 = set(task1.vocabulary)
    vocab2 = set(task2.vocabulary)
    return len(vocab1 & vocab2) / len(vocab1 | vocab2)

def task_similarity(task1, task2):
    """Combined similarity score"""
    label_sim = label_overlap(task1, task2)
    vocab_sim = vocabulary_overlap(task1, task2)
    return 0.5 * label_sim + 0.5 * vocab_sim
```

**Transfer success prediction**:
```python
from scipy.stats import spearmanr

# Build matrices
similarity_matrix = compute_pairwise_similarity(tasks)
transfer_matrix = compute_transfer_gains(results)

# Correlation between similarity and transfer
rho, p_value = spearmanr(
    similarity_matrix.flatten(),
    transfer_matrix.flatten()
)

# Report: ρ = X.XX, p < 0.05
# Interpretation: Higher task similarity → better transfer
```

---

## 6. Experimental Procedure

### 6.1 Timeline (6 Weeks with 7 Models)

**Week 1-2: Single-Task Baselines**
- Run 56 experiments (7 models × 8 tasks)
- Establish performance ceiling per task per model
- Identify top 3 best models (likely BlueBERT, PubMedBERT, Clinical-BERT)
- Generate Table 3 (model comparison)

**Week 3: Multi-Task Experiments (All Models)**
- Run 28 experiments (4 combinations × 7 models)
- Record total tokens per combination per model
- Evaluate on all task test sets
- Identify which models benefit most from MTL

**Week 4-5: Token-Controlled Baselines**
- **Priority**: Top 3 models → 96 experiments
- **If time**: All 7 models → 224 experiments
- This is the most important analysis (RQ2!)
- Critical for paper's main contribution

**Week 6: Analysis & Writing**
- Statistical tests (7 models × multiple comparisons)
- Model performance ranking
- Negative transfer analysis
- Task similarity correlation
- Generate all tables and figures
- Draft paper

**Alternative Fast Track (4 Weeks)** - If compute limited:
- Week 1: Single-task for ALL 7 models (56 experiments)
- Week 2: Multi-task for TOP 3 models only (12 experiments)
- Week 3: Token-controlled for TOP 3 models (96 experiments)
- Week 4: Analysis & writing

### 6.2 Computational Resources

**Platform**: Kaggle (free tier)
- GPU: NVIDIA T4 (16GB VRAM)
- Batch size: 32 (single-task), 16 (multi-task)
- Training time: 1.5-3 hours per experiment
- Total experiments: 56 (S1) + 28 (S2) + 96 (S3, top 3 models) = **180 experiments**
- Total GPU time: ~400 hours

**Compute strategy**:
- **Free tier** (Kaggle/Colab): 30 hours/week → 14 weeks total
- **Mixed** (mostly free + $50 paid): 6-8 weeks
- **Full paid** (Google Cloud T4): $140 total (400 hours × $0.35/hour)

**Recommended approach**:
- Use Kaggle free tier for S1 (single-task baselines)
- Use Google Cloud credits ($300 free) for S2-S3 (multi-task + token-controlled)
- Total cost: $0 (using free credits)

### 6.3 Reproducibility

**Random seeds**:
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

**Results storage**:
```json
{
  "experiment_id": "20260207_143022",
  "experiment_type": "single_task_baseline",
  "model": "dmis-lab/biobert-v1.1",
  "dataset": "bc2gm",
  "config": {...},
  "results": {
    "test_f1": 0.7834,
    "test_precision": 0.7912,
    "test_recall": 0.7758,
    "epochs_trained": 2.6,
    "total_tokens": 1234567,
    "training_time_seconds": 5400
  }
}
```

---

## 7. Expected Results

### 7.1 Research Questions (Expected Answers)

**RQ1: Model Comparison**
- Expected ranking: BlueBERT > PubMedBERT ≈ Clinical-BERT > BioMed-RoBERTa > BioBERT > RoBERTa > BERT
- BlueBERT wins on 6/8 tasks (hybrid pretraining advantage)
- Clinical-BERT best on clinical tasks (DDI)
- PubMedBERT best on biomedical research tasks (BC2GM, JNLPBA)

**RQ2: Architecture Analysis**
- RoBERTa-base beats BERT-base by +2-3% F1 (more robust training)
- BioMed-RoBERTa beats BioBERT by +1-2% F1 (architecture + domain)
- Conclusion: RoBERTa architecture provides consistent advantage

**RQ3: Pretraining Corpus Effect**
- General → Biomedical: +5-8% F1 (large gain)
- Biomedical → Clinical: +1-2% F1 (smaller gain)
- Biomedical + Clinical (BlueBERT): +2-3% over biomedical only
- Conclusion: Domain pretraining crucial, clinical data provides modest boost

**RQ4: Multi-Task Effectiveness**
- Expected: Multi-task improves F1 by +2-5% on 6/8 tasks
- All models benefit, but gains vary:
  - BERT-base: +5% (learns more from multi-task)
  - BlueBERT: +2% (already has strong representations)
- Negative transfer expected on 2/8 tasks (BIOSSES, HoC - very small/different)

**RQ5: Token-Controlled Analysis** ⭐ **KEY CONTRIBUTION**
- Expected: Multi-task still better by +2-3% even with equal tokens
- Holds across all 7 models (validates generalizability)
- Conclusion: Genuine cross-task transfer confirmed
- THIS IS THE PAPER'S MAIN CONTRIBUTION

**RQ6: Model-Specific Transfer**
- General models (BERT, RoBERTa) benefit MORE from MTL (+4-5%)
- Specialized models (BlueBERT, PubMedBERT) benefit LESS (+2-3%)
- Hypothesis: Specialized models already capture cross-task patterns in pretraining

**RQ7: Task Combinations**
- NER pair: Strong positive transfer (+4% across all models)
- RE pair: Moderate transfer (+2%)
- Mixed: Smaller gains (+1-2%)
- All 8: Potential negative transfer on small tasks
- Consistency: Transfer patterns similar across models

**RQ8: Practical Guidelines**
- **Model selection**:
  - General medical NLP: BlueBERT (best overall)
  - Biomedical research: PubMedBERT
  - Clinical applications: Clinical-BERT
  - Resource-constrained: BioBERT (lighter, still good)
- **Training strategy**:
  - Use multi-task when: Similar tasks, sufficient data (>5K samples each)
  - Single-task when: Tasks very different, tiny datasets (<500 samples)
  - General models benefit more from MTL than specialized models

### 7.2 Success Criteria

**Minimum for publication**:
- ✓ Token-controlled analysis shows MT > TC (p < 0.05)
- ✓ Effect size medium or large (Cohen's d > 0.5)
- ✓ Works on ≥5/8 datasets

**Strong paper**:
- ✓ MT > TC with +2-3% F1 average
- ✓ Large effect size (d > 0.8)
- ✓ Clear task similarity correlation
- ✓ Actionable guidelines for practitioners

---

## 8. Paper Structure

### 8.1 Sections (8 pages + references)

**Abstract** (1 paragraph)
- Problem: MTL studies conflate data exposure with transfer
- Method: Token-controlled baseline
- Results: +2.3% F1 with token parity
- Conclusion: Genuine transfer confirmed

**1. Introduction** (1 page)
- Medical NLP importance
- MTL promise but unclear gains
- Our contribution: Token-controlled analysis
- Paper roadmap

**2. Related Work** (0.75 pages)
- Multi-task learning (general)
- Medical MTL (BioBERT, Clinical BERT studies)
- Gap: No token-controlled comparisons

**3. Methodology** (1.5 pages)
- Datasets (Table 1)
- Models
- Training strategies (especially S3!)
- Token tracking implementation

**4. Experiments** (0.5 pages)
- Hyperparameters (Table 2)
- Computational setup
- Evaluation metrics

**5. Results** (2 pages)
- RQ1: Multi-task effectiveness (Table 3)
- RQ2: Token-controlled analysis (Table 4) ⭐ MAIN RESULT
- RQ3: Task combinations (Figure 1)
- RQ4: Negative transfer (Figure 2)

**6. Analysis** (1 page)
- Task similarity correlation
- Error analysis
- Practical implications

**7. Discussion** (0.75 pages)
- Why token control matters
- When to use MTL
- Limitations

**8. Conclusion** (0.5 pages)
- Token-controlled baseline (contribution)
- Genuine transfer confirmed
- Guidelines for practitioners
- Future work

**References** (2 pages)

### 8.2 Key Tables

**Table 1: Dataset Statistics**
| Dataset | Task | Samples | Tokens | Domain |
|---------|------|---------|--------|--------|
| ... | ... | ... | ... | ... |

**Table 2: Hyperparameters**
| Parameter | Value | Justification |
|-----------|-------|---------------|
| ... | ... | ... |

**Table 3: Single-Task Baseline Results (7 Models × 8 Tasks)**
| Dataset | BERT | RoBERTa | BioBERT | PubMedBERT | ClinicalBERT | BlueBERT | BioMed-RoBERTa | Best |
|---------|------|---------|---------|------------|--------------|----------|----------------|------|
| BC2GM | 0.72 | 0.74 | 0.78 | 0.79 | 0.76 | **0.81** | 0.78 | BlueBERT |
| JNLPBA | 0.68 | 0.71 | 0.75 | 0.77 | 0.74 | **0.78** | 0.76 | BlueBERT |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

**Table 4: Multi-Task vs Single-Task (Best Model)**
| Dataset | BlueBERT ST | BlueBERT MT | Gain | Significant? |
|---------|-------------|-------------|------|--------------|
| BC2GM | 0.81 | 0.84 | +3.7% | ✓ (p<0.01) |
| JNLPBA | 0.78 | 0.81 | +3.8% | ✓ (p<0.01) |
| ... | ... | ... | ... | ... |

**Table 4: Token-Controlled Analysis** ⭐ MOST IMPORTANT
| Dataset | ST Baseline | Multi-Task | Token-Controlled | Genuine Gain |
|---------|-------------|------------|------------------|--------------|
| BC2GM | 0.78 (1.2M) | 0.83 (7.6M) | 0.80 (7.6M) | +3% ✓ |
| ... | ... | ... | ... | ... |

### 8.3 Key Figures

**Figure 1: Task Combination Analysis**
- Heatmap: Which task pairs benefit most
- X-axis: Task 1, Y-axis: Task 2
- Color: Transfer gain

**Figure 2: Token-Controlled Comparison**
- Bar chart per dataset
- 3 bars: ST baseline, Multi-task, Token-controlled
- Show that MT > TC consistently

**Figure 3: Task Similarity vs Transfer**
- Scatter plot
- X-axis: Task similarity score
- Y-axis: Transfer gain
- Trendline + Spearman ρ

---

## 9. Limitations

### 9.1 Acknowledged Limitations

**Dataset limitations**:
- Limited to 8 publicly available datasets
- No access to proprietary clinical data
- English language only

**Model limitations**:
- BERT-family only (~110M parameters)
- Not testing larger models (GPT, Llama)
- No comparison to instruction-tuned models

**Methodological limitations**:
- Token counting is approximate (depends on tokenizer)
- Early stopping introduces variability
- Some test sets are small (<500 samples)

**Scope limitations**:
- Medical NLP only (generalization to other domains unknown)
- Classification tasks only (not generation)

### 9.2 Future Work

- Extend to larger models (Llama-3-8B with LoRA)
- Test on clinical note datasets (with IRB approval)
- Multi-lingual medical MTL
- Generation tasks (medical report summarization)

---

## 10. Ethical Considerations

### 10.1 Data Ethics

- All datasets publicly available
- No patient-identifiable information
- Proper attribution of data sources
- Reproducible experiments

### 10.2 Model Ethics

- Models are research prototypes
- NOT validated for clinical use
- Would require regulatory approval before deployment
- Bias assessment recommended before real-world use

### 10.3 Environmental Impact

- Using free/shared GPU resources
- Early stopping reduces unnecessary computation
- Estimated carbon footprint: ~5 kg CO2 (150 GPU hours on T4)

---

## 11. Novelty Statement

### 11.1 What IS Novel

✅ **Token-controlled baseline** (first in medical NLP)
✅ **Comprehensive model comparison** (7 BERT variants: first systematic comparison)
✅ **Architecture comparison** (BERT vs RoBERTa for medical NLP)
✅ **Pretraining corpus analysis** (general vs biomedical vs clinical)
✅ **Systematic 8-task analysis** (most comprehensive MTL study)
✅ **Negative transfer analysis** (task similarity correlation)
✅ **Practical guidelines** (evidence-based model selection + MTL recommendations)

### 11.2 What is NOT Novel (But Valid)

❌ Multi-task learning concept (well-established)
❌ BERT fine-tuning (standard practice)
❌ Individual medical NLP datasets (existing benchmarks)
❌ Domain-specific pretraining (BioBERT, Clinical-BERT exist)

### 11.3 Contribution Summary

**Technical contributions**:
1. Token-controlled baseline methodology
2. Fair multi-model comparison protocol

**Scientific contributions**:
1. Separating data exposure from genuine transfer
2. Quantifying pretraining corpus effect (general → biomedical → clinical)
3. BERT vs RoBERTa architecture comparison for medical tasks

**Practical contributions**:
1. Model selection guide (which BERT variant for which task type?)
2. MTL task combination guidelines
3. Cost-benefit analysis (is expensive clinical pretraining worth it?)

---

## 12. Publication Strategy

### 12.1 Target Venues (Ranked by Fit)

**Tier 1 (Best Fit)**:
1. **BioNLP Workshop** (at ACL/EMNLP)
   - Perfect domain fit
   - Values rigorous medical NLP work
   - Acceptance rate: ~35%

2. **EMNLP Findings**
   - Good for solid empirical work
   - Less novel than main conference, but rigorous
   - Acceptance rate: ~25%

**Tier 2 (Good Fit)**:
3. **Journal of Biomedical Informatics**
   - Values comprehensive experiments
   - No page limits (can include more analysis)
   - Impact factor: 4.0

4. **Clinical NLP Workshop**
   - Domain-specific venue
   - Values practical contributions

### 12.2 Submission Timeline

- **Week 1-3**: Experiments
- **Week 4**: Draft paper
- **Week 5**: Internal review, revisions
- **Week 6**: Submit to BioNLP Workshop (next deadline)

### 12.3 Reviewer Anticipation

**Expected questions**:
1. "Why not test larger models?"
   → Answer: Focus on deployable models, established baselines

2. "Why BERT when GPT exists?"
   → Answer: Fair comparison to prior work, practical constraints

3. "Only 8 datasets?"
   → Answer: Most comprehensive medical MTL study to date

4. "Token control seems obvious?"
   → Answer: Never done before in medical NLP literature

---

## Summary

**Research Question**: Does multi-task learning provide genuine cross-task transfer or just benefit from increased data exposure?

**Method**: Token-controlled baseline that equalizes data exposure

**Expected Result**: Multi-task beats token-controlled single-task by +2-3% F1

**Contribution**: First rigorous separation of data exposure vs transfer in medical NLP

**Publication Target**: BioNLP Workshop or EMNLP Findings

**Novelty**: Sufficient for mid-tier venue, good thesis chapter

---

**Document Version**: 1.0 (BERT-focused)
**Last Updated**: 2026-02-07
**Status**: Ready for implementation
**Estimated Project Duration**: 4 weeks
**Estimated Cost**: $0 (Kaggle free tier)
