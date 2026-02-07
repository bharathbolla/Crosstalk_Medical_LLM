# src/evaluation/CLAUDE.md — Evaluation Protocol

## Evaluation Philosophy

This project must survive reviewer scrutiny at NAACL/EMNLP. Every claim needs statistical backing. Every comparison needs controls.

## Metrics by Task

| Task | Primary | Secondary | Eval Tool | Match Criteria |
|---|---|---|---|---|
| 2014-T7 | Strict F1 | Relaxed F1, Accuracy | Official eval.py | Exact span + CUI |
| 2015-T14 | Strict F1 | Relaxed F1 | Official eval.py | Span match (discont.) |
| 2016-T12 | F1 per subtask | Precision, Recall | Official eval.py | Span + type + relation |
| 2017-T3 | MAP | MRR, P@5 | trec_eval | Ranking correctness |
| 2021-T6 | Micro F1 | Macro F1, per-class | Official eval.py | Exact span + relation |

## Aggregate Score

Normalized aggregate = mean of (task_score / best_known_score) across tasks.

**WARNING**: Aggregate can hide tradeoffs. Always report per-task results alongside.

Add per-task win/tie/loss counts:
```python
def wins_ties_losses(method_a_scores, method_b_scores, threshold=0.5):
    """Count tasks where A wins/ties/loses vs B."""
    wins = sum(a > b + threshold for a, b in zip(method_a_scores, method_b_scores))
    losses = sum(b > a + threshold for a, b in zip(method_a_scores, method_b_scores))
    ties = len(method_a_scores) - wins - losses
    return wins, ties, losses
```

## Statistical Tests

### Bootstrap Confidence Intervals (95%)
```python
def bootstrap_ci(scores, n_bootstrap=10000, ci=0.95):
    """Bootstrap 95% CI for any metric."""
    boot_scores = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        boot_scores.append(np.mean(sample))
    lower = np.percentile(boot_scores, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_scores, (1 + ci) / 2 * 100)
    return lower, upper
```

### Paired Permutation Test
```python
def paired_permutation_test(scores_a, scores_b, n_permutations=10000):
    """Test if difference between two systems is significant."""
    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    count = 0
    for _ in range(n_permutations):
        mask = np.random.binomial(1, 0.5, size=len(scores_a))
        perm_a = np.where(mask, scores_a, scores_b)
        perm_b = np.where(mask, scores_b, scores_a)
        if np.mean(perm_a) - np.mean(perm_b) >= observed_diff:
            count += 1
    return count / n_permutations
```

### Required Comparisons (ALL need significance tests)
1. S2 vs S1 (multi-task vs single-task)
2. S3a vs S1 (shared-private vs single-task)
3. S3b vs S3a (hierarchical vs flat)
4. S3b vs token-controlled S1 (**most important** — RQ5)
5. A4 vs A1, A2, A3 (architecture ablations)
6. S5 (quantized) vs S3a/S3b (full precision)

## Contamination Check Protocol (contamination.py)

Three-layer detection, run in Phase 0:

### Layer 1: Zero-Shot Audit
- Run base model (no fine-tuning) on test set
- If zero-shot F1 > 70% of published SOTA → flag as contaminated
- Record all prompts for reproducibility

### Layer 2: N-gram Overlap
- Generate 100-token continuations from test set prefixes
- Measure verbatim 8/10/13-gram overlap
- Threshold: >5% of samples with 13-gram matches → contaminated

### Layer 3: Min-K% Probing
- Compare test set log-likelihood vs control clinical text
- Mann-Whitney U test, p < 0.01 → contaminated

## Probing Tasks (probing.py)

After training, freeze shared adapter and train linear probes:

| Probe | Task | Labels | Source | Purpose |
|---|---|---|---|---|
| Medical Concept Type | Classify token as Drug/Symptom/Procedure/Anatomy/Other | 5 classes | UMLS types | Does adapter learn medical ontology? |
| Negation Detection | Detect negated entities ("no fever" vs "fever") | Binary | NegEx patterns | Critical for clinical safety |
| Abbreviation Expansion | Match abbreviation to full form | Accuracy | CASI dataset | Common clinical challenge |
| Temporal Ordering | Predict temporal order of two events | 3 classes | TimeML | Tests reasoning capability |

## Calibration (calibration.py)

ECE (Expected Calibration Error) — measures if model confidence matches accuracy.

```python
def expected_calibration_error(confidences, predictions, labels, n_bins=15):
    """Lower is better. Medical models MUST know when they're uncertain."""
```

Report ECE for top 3 model configurations only (to save compute).
Generate reliability diagrams (calibration plots) for the paper.

## Error Analysis (error_analysis.py)

6-category error taxonomy:

| Category | Description | How to Detect | Affected Tasks |
|---|---|---|---|
| Abbreviation errors | Fails on "MI", "CHF", "DVT" | Match against abbreviation list | All NER |
| Negation scope | Wrong polarity on negated entities | NegEx pattern matching | 2014-T7, 2015-T14 |
| Discontiguous spans | Misses linked spans | Compare span structure | 2015-T14 |
| Rare entities | Fails on low-frequency types | Split by entity frequency | All NER |
| Temporal ambiguity | Wrong temporal relations for implicit time | Manual annotation of errors | 2016-T12 |
| Cross-sentence refs | Misses cross-sentence entities/relations | Check sentence boundary overlap | 2021-T6, 2016-T12 |

## Transfer Analysis (transfer_analysis.py)

### Transfer Matrix
For each pair of tasks (A, B): measure performance on B when also trained on A.
Produces 5×5 heatmap showing positive/negative transfer.

### Task Similarity Metrics (for RQ4)
```python
def label_schema_similarity(task_a, task_b):
    """Measure overlap in label schemas (BIO, relation types, etc.)"""
    
def vocabulary_overlap(task_a_data, task_b_data, tokenizer):
    """Measure token-level vocabulary overlap between tasks."""
    
def predict_transfer(similarity_scores, transfer_deltas):
    """Correlate task similarity with transfer success."""
    # Spearman correlation between similarity and transfer delta
    # This tests RQ4 empirically
```

### Negative Transfer Detection
```python
def detect_negative_transfer(single_task_scores, multi_task_scores):
    """
    For each task, check if multi-task HURTS performance.
    Negative transfer = multi_task_score < single_task_score - noise_margin
    """
```

## GPT-4 Framing

GPT-4o-mini results are **reference points, not baselines**.

In the paper, explicitly state:
> "We do not claim superiority over GPT-4; we quantify the efficiency gap under deployment-realistic constraints where patient data cannot leave the hospital network."

Add both zero-shot and 5-shot results for credibility.

## Paper Figures to Generate

| Figure | Script | Data Source |
|---|---|---|
| Transfer heatmap (5×5) | notebooks/03_transfer_heatmap.ipynb | transfer_analysis.py |
| Pareto frontier (cost vs perf) | notebooks/04_pareto_frontier.ipynb | efficiency + metrics |
| Probing accuracy by adapter | notebooks/05_probing_results.ipynb | probing.py |
| Reliability diagram (ECE) | notebooks/06_calibration_plots.ipynb | calibration.py |
| Error distribution (6 categories) | notebooks/07_error_analysis.ipynb | error_analysis.py |
| Token-controlled comparison | notebooks/08_token_controlled.ipynb | RQ5 results |
| Task-wise radar plot | generate_paper_tables.py | all results |
| Architecture ablation (A1-A4) | generate_paper_tables.py | ablation results |
