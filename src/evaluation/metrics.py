"""Task-specific metrics and statistical tests.

Implements:
- Task-specific metric computation (F1, MAP, etc.)
- Bootstrap confidence intervals
- Paired permutation tests
- Win/tie/loss counts
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import Counter

try:
    from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
    from seqeval.scheme import IOB2
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False

try:
    from sklearn.metrics import f1_score as sklearn_f1, precision_recall_fscore_support
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def compute_task_metrics(
    task_name: str,
    predictions: Any,
    labels: Any,
    **kwargs,
) -> Dict[str, float]:
    """Compute task-specific metrics.

    Dispatches to correct metric function based on task name.

    Args:
        task_name: Name of the task
        predictions: Model predictions (format depends on task)
        labels: Ground truth labels
        **kwargs: Additional task-specific arguments

    Returns:
        Dictionary of metrics

    Example:
        >>> metrics = compute_task_metrics(
        ...     "semeval2014t7",
        ...     predictions=pred_tags,
        ...     labels=true_tags
        ... )
        >>> print(metrics["strict_f1"])
    """
    # Map task to metric function
    if task_name in ["semeval2014t7", "semeval2021t6_level1"]:
        # NER tasks: BIO tagging
        return compute_ner_metrics(predictions, labels)

    elif task_name == "semeval2015t14":
        # Discontiguous span task
        return compute_span_metrics(predictions, labels, **kwargs)

    elif task_name == "semeval2016t12":
        # Temporal relation extraction
        return compute_relation_metrics(predictions, labels)

    elif task_name == "semeval2017t3":
        # QA ranking
        return compute_ranking_metrics(predictions, labels, **kwargs)

    elif task_name == "semeval2021t6_level2":
        # Relation extraction
        return compute_relation_metrics(predictions, labels)

    else:
        raise ValueError(f"Unknown task: {task_name}")


def compute_ner_metrics(
    predictions: List[List[str]],
    labels: List[List[str]],
) -> Dict[str, float]:
    """Compute NER metrics (F1, precision, recall).

    Uses seqeval for span-level evaluation.

    Args:
        predictions: List of predicted BIO tag sequences
        labels: List of gold BIO tag sequences

    Returns:
        Dictionary with strict_f1, precision, recall, accuracy
    """
    if not SEQEVAL_AVAILABLE:
        raise ImportError("seqeval is required for NER metrics. Install with: pip install seqeval")

    # Compute strict F1 (exact span match)
    strict_f1 = f1_score(labels, predictions, mode='strict', scheme=IOB2)
    precision = precision_score(labels, predictions, mode='strict', scheme=IOB2)
    recall = recall_score(labels, predictions, mode='strict', scheme=IOB2)

    # Compute relaxed F1 (type-only match)
    relaxed_f1 = f1_score(labels, predictions, mode='relaxed', scheme=IOB2)

    # Token-level accuracy
    correct = sum(
        sum(p == l for p, l in zip(pred_seq, label_seq))
        for pred_seq, label_seq in zip(predictions, labels)
    )
    total = sum(len(seq) for seq in labels)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "strict_f1": strict_f1,
        "relaxed_f1": relaxed_f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "primary_metric": strict_f1,  # Primary metric for comparison
    }


def compute_span_metrics(
    predictions: List[List[Tuple[int, int, str]]],
    labels: List[List[Tuple[int, int, str]]],
    allow_partial: bool = False,
) -> Dict[str, float]:
    """Compute span-level metrics for discontiguous entities.

    Args:
        predictions: List of predicted spans (start, end, label)
        labels: List of gold spans
        allow_partial: Whether to allow partial matches

    Returns:
        Dictionary with strict_f1, partial_f1, precision, recall
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    partial_matches = 0

    for pred_spans, gold_spans in zip(predictions, labels):
        # Convert to sets for easier matching
        pred_set = set(pred_spans)
        gold_set = set(gold_spans)

        # Exact matches
        exact_matches = pred_set & gold_set
        true_positives += len(exact_matches)

        # False positives
        false_positives += len(pred_set - gold_set)

        # False negatives
        false_negatives += len(gold_set - pred_set)

        # Partial matches (if allowed)
        if allow_partial:
            for pred_span in pred_set - gold_set:
                pred_start, pred_end, pred_label = pred_span
                for gold_start, gold_end, gold_label in gold_set - pred_set:
                    # Check for overlap
                    if pred_label == gold_label:
                        if (pred_start <= gold_end and pred_end >= gold_start):
                            partial_matches += 1
                            break

    # Compute metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    strict_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "strict_f1": strict_f1,
        "precision": precision,
        "recall": recall,
        "primary_metric": strict_f1,
    }

    if allow_partial:
        partial_f1 = 2 * (true_positives + partial_matches) / (
            2 * true_positives + partial_matches + false_positives + false_negatives
        )
        metrics["partial_f1"] = partial_f1

    return metrics


def compute_relation_metrics(
    predictions: List[List[Tuple]],
    labels: List[List[Tuple]],
) -> Dict[str, float]:
    """Compute relation extraction metrics.

    Args:
        predictions: List of predicted relations (head, tail, relation_type)
        labels: List of gold relations

    Returns:
        Dictionary with micro_f1, macro_f1, precision, recall
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required for RE metrics")

    # Flatten and align predictions and labels
    all_preds = []
    all_labels = []

    for pred_rels, gold_rels in zip(predictions, labels):
        # Exact match: (head, tail, relation) must all match
        pred_set = set(pred_rels)
        gold_set = set(gold_rels)

        # True positives
        tp = len(pred_set & gold_set)

        # False positives
        fp = len(pred_set - gold_set)

        # False negatives
        fn = len(gold_set - pred_set)

        all_preds.extend([1] * tp + [1] * fp)
        all_labels.extend([1] * tp + [0] * fp + [1] * fn)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )

    return {
        "micro_f1": f1,
        "precision": precision,
        "recall": recall,
        "primary_metric": f1,
    }


def compute_ranking_metrics(
    predictions: List[List[float]],
    labels: List[List[int]],
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, float]:
    """Compute ranking metrics for QA.

    Args:
        predictions: List of relevance scores per sample
        labels: List of relevance labels (0/1/2 for Bad/Useful/Good)
        k_values: K values for Precision@K

    Returns:
        Dictionary with MAP, MRR, P@K metrics
    """
    # Mean Average Precision (MAP)
    average_precisions = []

    for scores, rels in zip(predictions, labels):
        # Sort by scores
        sorted_indices = np.argsort(scores)[::-1]
        sorted_rels = [rels[i] for i in sorted_indices]

        # Compute average precision
        relevant_count = 0
        precision_sum = 0.0

        for i, rel in enumerate(sorted_rels):
            if rel > 0:  # Relevant (PotentiallyUseful or Good)
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i

        if relevant_count > 0:
            ap = precision_sum / relevant_count
        else:
            ap = 0.0

        average_precisions.append(ap)

    map_score = np.mean(average_precisions)

    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []

    for scores, rels in zip(predictions, labels):
        sorted_indices = np.argsort(scores)[::-1]
        sorted_rels = [rels[i] for i in sorted_indices]

        # Find first relevant item
        for i, rel in enumerate(sorted_rels):
            if rel > 0:
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)

    mrr = np.mean(reciprocal_ranks)

    # Precision@K
    precision_at_k = {}
    for k in k_values:
        precisions = []
        for scores, rels in zip(predictions, labels):
            sorted_indices = np.argsort(scores)[::-1][:k]
            relevant_in_top_k = sum(rels[i] > 0 for i in sorted_indices)
            precisions.append(relevant_in_top_k / k)

        precision_at_k[f"p@{k}"] = np.mean(precisions)

    return {
        "map": map_score,
        "mrr": mrr,
        **precision_at_k,
        "primary_metric": map_score,
    }


def bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    statistic: str = "mean",
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval.

    Args:
        scores: Array of scores
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default: 0.95 for 95% CI)
        statistic: Statistic to compute ("mean" or "median")

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(scores, size=len(scores), replace=True)

        # Compute statistic
        if statistic == "mean":
            stat = np.mean(sample)
        elif statistic == "median":
            stat = np.median(sample)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

        bootstrap_stats.append(stat)

    # Compute percentiles
    alpha = 1 - ci
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return lower, upper


def paired_permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 10000,
    alternative: str = "two-sided",
) -> float:
    """Paired permutation test for significance.

    Tests if difference between two systems is statistically significant.

    Args:
        scores_a: Scores for system A
        scores_b: Scores for system B
        n_permutations: Number of permutations
        alternative: "two-sided", "greater", or "less"

    Returns:
        P-value
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have same length")

    # Observed difference
    observed_diff = np.mean(scores_a) - np.mean(scores_b)

    # Permutation test
    count = 0

    for _ in range(n_permutations):
        # Random permutation
        mask = np.random.binomial(1, 0.5, size=len(scores_a))

        # Swap scores where mask == 1
        perm_a = np.where(mask, scores_a, scores_b)
        perm_b = np.where(mask, scores_b, scores_a)

        # Compute difference
        perm_diff = np.mean(perm_a) - np.mean(perm_b)

        # Count extreme differences
        if alternative == "two-sided":
            if abs(perm_diff) >= abs(observed_diff):
                count += 1
        elif alternative == "greater":
            if perm_diff >= observed_diff:
                count += 1
        elif alternative == "less":
            if perm_diff <= observed_diff:
                count += 1

    p_value = count / n_permutations

    return p_value


def wins_ties_losses(
    scores_a: List[float],
    scores_b: List[float],
    threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """Count wins, ties, and losses between two methods.

    Args:
        scores_a: Scores for method A (per task)
        scores_b: Scores for method B (per task)
        threshold: Difference threshold for tie (e.g., 0.5 F1 points)

    Returns:
        Tuple of (wins, ties, losses) for method A

    Example:
        >>> scores_a = [85.2, 72.1, 68.5, 80.3, 75.8]
        >>> scores_b = [83.1, 72.8, 70.2, 79.5, 74.2]
        >>> wins, ties, losses = wins_ties_losses(scores_a, scores_b)
        >>> print(f"A vs B: {wins}W / {ties}T / {losses}L")
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length")

    wins = 0
    ties = 0
    losses = 0

    for a, b in zip(scores_a, scores_b):
        diff = a - b

        if diff > threshold:
            wins += 1
        elif diff < -threshold:
            losses += 1
        else:
            ties += 1

    return wins, ties, losses
