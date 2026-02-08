"""
COMPLETE FIXED METRICS
Automatic metrics computation for all 8 task types

This replaces the compute_metrics function in Cell 10/12 in your notebook
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error
)
from seqeval.metrics import (
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall
)

# ============================================
# TASK-SPECIFIC METRICS
# ============================================

def compute_ner_metrics(predictions, labels, label_list):
    """
    Compute metrics for NER tasks (BC2GM, JNLPBA).
    Uses seqeval for sequence labeling evaluation.
    """

    # Convert predictions to label indices
    predictions = np.argmax(predictions, axis=2)

    # Remove padding (-100 labels)
    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels):
        true_label = []
        true_pred = []

        for p, l in zip(prediction, label):
            if l != -100:  # Skip padding
                # Ensure indices are within bounds
                if l < len(label_list):
                    true_label.append(label_list[l])
                else:
                    true_label.append("O")

                if p < len(label_list):
                    true_pred.append(label_list[p])
                else:
                    true_pred.append("O")

        if true_label:  # Only add if non-empty
            true_labels.append(true_label)
            true_predictions.append(true_pred)

    # Calculate NER metrics using seqeval
    try:
        f1 = seqeval_f1(true_labels, true_predictions)
        precision = seqeval_precision(true_labels, true_predictions)
        recall = seqeval_recall(true_labels, true_predictions)
    except Exception as e:
        print(f"⚠️  Metrics calculation warning: {e}")
        f1, precision, recall = 0.0, 0.0, 0.0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def compute_classification_metrics(predictions, labels):
    """
    Compute metrics for classification tasks (RE, GAD, PubMedQA).
    Uses sklearn for standard classification metrics.
    """

    # Get predicted class
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    try:
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)
        accuracy = accuracy_score(labels, predictions)
    except Exception as e:
        print(f"⚠️  Metrics calculation warning: {e}")
        f1, precision, recall, accuracy = 0.0, 0.0, 0.0, 0.0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
    }


def compute_multilabel_metrics(predictions, labels):
    """
    Compute metrics for multi-label classification (HoC).
    """

    # Apply sigmoid to convert raw logits to probabilities, then threshold
    probs = 1 / (1 + np.exp(-predictions))  # sigmoid
    predictions = (probs > 0.3).astype(int)  # Lower threshold for imbalanced data

    # Calculate metrics per label, then average
    try:
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        precision = precision_score(labels, predictions, average='macro', zero_division=0)
        recall = recall_score(labels, predictions, average='macro', zero_division=0)

        # Exact match ratio (all labels must match)
        exact_match = accuracy_score(labels, predictions)
    except Exception as e:
        print(f"⚠️  Metrics calculation warning: {e}")
        f1, precision, recall, exact_match = 0.0, 0.0, 0.0, 0.0

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'exact_match': exact_match,
    }


def compute_regression_metrics(predictions, labels):
    """
    Compute metrics for regression tasks (BIOSSES).
    """

    # Flatten predictions
    predictions = predictions.squeeze()

    try:
        # Regression metrics
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(labels, predictions)

        # Pearson correlation
        from scipy.stats import pearsonr, spearmanr
        pearson_r, pearson_p = pearsonr(predictions, labels)
        spearman_r, spearman_p = spearmanr(predictions, labels)
    except Exception as e:
        print(f"⚠️  Metrics calculation warning: {e}")
        mse, rmse, mae = 0.0, 0.0, 0.0
        pearson_r, spearman_r = 0.0, 0.0

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'pearson_r': pearson_r,
        'spearman_r': spearman_r,
    }


# ============================================
# UNIVERSAL METRICS FUNCTION
# ============================================

def compute_metrics_universal(pred, task_name, task_config):
    """
    Universal metrics function that routes to appropriate metric calculator.

    Args:
        pred: Predictions from trainer (EvalPrediction object)
        task_name: Dataset name (e.g., 'bc2gm')
        task_config: Task configuration dict

    Returns:
        dict: Metrics appropriate for task type
    """

    predictions = pred.predictions
    labels = pred.label_ids

    task_type = task_config.get('task_type', 'unknown')
    label_list = task_config.get('labels', [])

    # Route to appropriate metrics function
    if task_type == 'ner':
        return compute_ner_metrics(predictions, labels, label_list)

    elif task_type in ['re', 'classification', 'qa']:
        return compute_classification_metrics(predictions, labels)

    elif task_type == 'multilabel_classification':
        return compute_multilabel_metrics(predictions, labels)

    elif task_type == 'similarity':
        return compute_regression_metrics(predictions, labels)

    else:
        print(f"⚠️  Unknown task type: {task_type}, using classification metrics")
        return compute_classification_metrics(predictions, labels)


# ============================================
# COMPLETE CELL 10/12 CODE (for notebook)
# ============================================

CELL_METRICS_CODE = '''
# ============================================
# CELL 10: Metrics Computation (Universal)
# ============================================

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from seqeval.metrics import f1_score as seqeval_f1, precision_score as seqeval_precision, recall_score as seqeval_recall

def compute_metrics(pred):
    """
    Universal metrics function that auto-detects task type.
    Works for all 8 datasets!
    """

    predictions = pred.predictions
    labels = pred.label_ids

    # Get task info (task_name should be set globally or passed via trainer)
    # For single-task, use primary dataset
    task_name = CONFIG['datasets'][0]
    task_stats = dataset_stats[task_name]
    task_type = task_stats['task_type']

    # NER tasks (BC2GM, JNLPBA)
    if task_type == 'ner':
        # Get label list from task config
        from COMPLETE_FIXED_DATASET import TASK_CONFIGS
        label_list = TASK_CONFIGS[task_name]['labels']

        # Convert predictions
        predictions = np.argmax(predictions, axis=2)

        # Remove padding
        true_labels = []
        true_predictions = []

        for prediction, label in zip(predictions, labels):
            true_label = []
            true_pred = []

            for p, l in zip(prediction, label):
                if l != -100:
                    if l < len(label_list):
                        true_label.append(label_list[l])
                    else:
                        true_label.append("O")

                    if p < len(label_list):
                        true_pred.append(label_list[p])
                    else:
                        true_pred.append("O")

            if true_label:
                true_labels.append(true_label)
                true_predictions.append(true_pred)

        # Calculate NER metrics
        try:
            f1 = seqeval_f1(true_labels, true_predictions)
            precision = seqeval_precision(true_labels, true_predictions)
            recall = seqeval_recall(true_labels, true_predictions)
        except Exception as e:
            print(f"⚠️  Warning: {e}")
            f1, precision, recall = 0.0, 0.0, 0.0

        return {'f1': f1, 'precision': precision, 'recall': recall}

    # Classification tasks (RE, GAD, QA)
    elif task_type in ['re', 'classification', 'qa']:
        predictions = np.argmax(predictions, axis=1)

        try:
            f1 = f1_score(labels, predictions, average='macro', zero_division=0)
            precision = precision_score(labels, predictions, average='macro', zero_division=0)
            recall = recall_score(labels, predictions, average='macro', zero_division=0)
            accuracy = accuracy_score(labels, predictions)
        except Exception as e:
            print(f"⚠️  Warning: {e}")
            f1, precision, recall, accuracy = 0.0, 0.0, 0.0, 0.0

        return {'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy}

    # Multi-label classification (HoC)
    elif task_type == 'multilabel_classification':
        predictions = (predictions > 0.5).astype(int)

        try:
            f1 = f1_score(labels, predictions, average='macro', zero_division=0)
            precision = precision_score(labels, predictions, average='macro', zero_division=0)
            recall = recall_score(labels, predictions, average='macro', zero_division=0)
        except Exception as e:
            print(f"⚠️  Warning: {e}")
            f1, precision, recall = 0.0, 0.0, 0.0

        return {'f1': f1, 'precision': precision, 'recall': recall}

    # Regression (BIOSSES)
    elif task_type == 'similarity':
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from scipy.stats import pearsonr, spearmanr

        predictions = predictions.squeeze()

        try:
            mse = mean_squared_error(labels, predictions)
            rmse = np.sqrt(mse)
            pearson_r, _ = pearsonr(predictions, labels)
            spearman_r, _ = spearmanr(predictions, labels)
        except Exception as e:
            print(f"⚠️  Warning: {e}")
            mse, rmse, pearson_r, spearman_r = 0.0, 0.0, 0.0, 0.0

        return {'mse': mse, 'rmse': rmse, 'pearson_r': pearson_r, 'spearman_r': spearman_r}

    else:
        # Default to classification
        predictions = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        return {'f1': f1}


print("✅ Metrics function loaded (supports all 8 task types)")
'''

if __name__ == "__main__":
    print("="*60)
    print("METRICS CODE FOR ALL 8 TASKS")
    print("="*60)
    print("\nThis code automatically computes:")
    print("  - NER: F1, Precision, Recall (seqeval)")
    print("  - RE/Classification/QA: F1, Precision, Recall, Accuracy (sklearn)")
    print("  - Multi-label: F1, Precision, Recall, Exact Match")
    print("  - Regression: MSE, RMSE, MAE, Pearson/Spearman correlation")
    print("\nCopy CELL_METRICS_CODE to your notebook Cell 10!")
    print("="*60)
