"""Error analysis with 6-category taxonomy.

Categorizes errors to understand model failures:
1. Abbreviation errors
2. Negation scope errors
3. Discontiguous span errors
4. Rare entity errors
5. Temporal ambiguity errors
6. Cross-sentence reference errors
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter


# Error categories
ERROR_CATEGORIES = [
    "abbreviation",
    "negation_scope",
    "discontiguous",
    "rare_entity",
    "temporal_ambiguity",
    "cross_sentence",
]


def categorize_errors(
    predictions: List[Any],
    labels: List[Any],
    task_config: Dict[str, Any],
    texts: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Categorize errors into 6 categories.

    Args:
        predictions: Model predictions
        labels: Ground truth labels
        task_config: Task configuration
        texts: Original texts (optional, for context)

    Returns:
        Dictionary mapping category to error count
    """
    error_counts = {cat: 0 for cat in ERROR_CATEGORIES}

    # TODO: Implement error categorization
    # Requires:
    # 1. Identify errors (predictions != labels)
    # 2. For each error, determine category:
    #    - Check if involves abbreviation
    #    - Check for negation patterns
    #    - Check if spans are discontiguous
    #    - Check entity frequency
    #    - Check temporal markers
    #    - Check sentence boundaries

    return error_counts


def is_abbreviation_error(
    pred_span: str,
    gold_span: str,
    abbr_list: Set[str],
) -> bool:
    """Check if error involves medical abbreviation.

    Args:
        pred_span: Predicted span text
        gold_span: Gold span text
        abbr_list: Set of known medical abbreviations

    Returns:
        True if error involves abbreviation
    """
    # Check if either span contains abbreviation
    pred_tokens = pred_span.upper().split()
    gold_tokens = gold_span.upper().split()

    return any(token in abbr_list for token in pred_tokens + gold_tokens)


def is_negation_error(
    pred_span: Tuple[int, int, str],
    gold_span: Tuple[int, int, str],
    text: str,
) -> bool:
    """Check if error involves negation scope.

    Args:
        pred_span: (start, end, label)
        gold_span: (start, end, label)
        text: Full text

    Returns:
        True if error involves negation
    """
    # Common negation cues
    negation_cues = ["no", "not", "without", "absent", "denies", "negative"]

    # Check context around spans
    start = max(0, min(pred_span[0], gold_span[0]) - 50)
    end = max(pred_span[1], gold_span[1]) + 50

    context = text[start:end].lower()

    return any(cue in context for cue in negation_cues)


# Abbreviation list (partial - for demonstration)
MEDICAL_ABBREVIATIONS = {
    "MI", "CHF", "DVT", "PE", "HTN", "DM", "COPD", "CABG",
    "PTCA", "EKG", "ECG", "CT", "MRI", "BP", "HR", "RR",
}
