"""Transfer analysis for RQ4: negative transfer detection.

Analyzes:
- Transfer matrix (5×5 heatmap)
- Task similarity metrics
- Negative transfer detection
- Correlation between similarity and transfer success
"""

from typing import Dict, List, Tuple, Set
import numpy as np
from scipy import stats


def compute_transfer_matrix(
    single_task_results: Dict[str, Dict[str, float]],
    multi_task_results: Dict[str, Dict[str, float]],
) -> np.ndarray:
    """Compute 5×5 transfer matrix for heatmap.

    Matrix[i, j] = performance on task j when also trained on task i

    Args:
        single_task_results: {task: {metric: score}}
        multi_task_results: {task: {metric: score}}

    Returns:
        5×5 numpy array of transfer deltas
    """
    tasks = list(single_task_results.keys())
    n_tasks = len(tasks)

    # Initialize transfer matrix
    transfer_matrix = np.zeros((n_tasks, n_tasks))

    for i, task_i in enumerate(tasks):
        for j, task_j in enumerate(tasks):
            if i == j:
                # Diagonal: no transfer (self)
                transfer_matrix[i, j] = 0.0
            else:
                # Transfer delta: multi-task score - single-task score
                st_score = single_task_results[task_j].get("primary_metric", 0.0)
                mt_score = multi_task_results[task_j].get("primary_metric", 0.0)
                transfer_matrix[i, j] = mt_score - st_score

    return transfer_matrix


def label_schema_similarity(
    task_a: str,
    task_b: str,
    task_configs: Dict[str, Dict],
) -> float:
    """Measure overlap in label schemas.

    Args:
        task_a: First task name
        task_b: Second task name
        task_configs: Task configurations with label schemas

    Returns:
        Jaccard similarity of label sets
    """
    labels_a = set(task_configs[task_a].get("labels", []))
    labels_b = set(task_configs[task_b].get("labels", []))

    if len(labels_a) == 0 or len(labels_b) == 0:
        return 0.0

    intersection = len(labels_a & labels_b)
    union = len(labels_a | labels_b)

    return intersection / union if union > 0 else 0.0


def vocabulary_overlap(
    task_a_data: List[str],
    task_b_data: List[str],
    tokenizer,
) -> float:
    """Measure token-level vocabulary overlap.

    Args:
        task_a_data: List of texts from task A
        task_b_data: List of texts from task B
        tokenizer: Tokenizer

    Returns:
        Jaccard similarity of vocabularies
    """
    # Tokenize and get vocabulary
    vocab_a = set()
    for text in task_a_data:
        tokens = tokenizer.tokenize(text)
        vocab_a.update(tokens)

    vocab_b = set()
    for text in task_b_data:
        tokens = tokenizer.tokenize(text)
        vocab_b.update(tokens)

    # Jaccard similarity
    intersection = len(vocab_a & vocab_b)
    union = len(vocab_a | vocab_b)

    return intersection / union if union > 0 else 0.0


def detect_negative_transfer(
    single_task_scores: Dict[str, float],
    multi_task_scores: Dict[str, float],
    noise_margin: float = 0.5,
) -> List[str]:
    """Detect tasks where multi-task hurts performance.

    Args:
        single_task_scores: {task: score}
        multi_task_scores: {task: score}
        noise_margin: Threshold for significance (F1 points)

    Returns:
        List of tasks with negative transfer
    """
    negative_transfer_tasks = []

    for task in single_task_scores.keys():
        st_score = single_task_scores[task]
        mt_score = multi_task_scores.get(task, 0.0)

        # Negative transfer if multi-task significantly worse
        if mt_score < st_score - noise_margin:
            negative_transfer_tasks.append(task)

    return negative_transfer_tasks


def predict_transfer_success(
    similarity_matrix: np.ndarray,
    transfer_matrix: np.ndarray,
) -> Tuple[float, float]:
    """Correlate task similarity with transfer success.

    Tests RQ4: Can we predict negative transfer from task similarity?

    Args:
        similarity_matrix: n×n matrix of task similarities
        transfer_matrix: n×n matrix of transfer deltas

    Returns:
        Tuple of (spearman_rho, p_value)
    """
    # Flatten matrices (exclude diagonal)
    n = similarity_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)

    similarities_flat = similarity_matrix[mask]
    transfers_flat = transfer_matrix[mask]

    # Spearman correlation
    rho, p_value = stats.spearmanr(similarities_flat, transfers_flat)

    return rho, p_value
