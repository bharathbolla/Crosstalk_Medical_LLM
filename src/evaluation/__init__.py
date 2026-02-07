"""Evaluation modules for metrics, contamination, calibration, and analysis."""

from .metrics import (
    compute_task_metrics,
    bootstrap_ci,
    paired_permutation_test,
    wins_ties_losses,
)
from .contamination import ContaminationChecker, ContaminationResult
from .calibration import (
    expected_calibration_error,
    plot_reliability_diagram,
)
from .probing import LinearProbe, evaluate_adapter_probes
from .transfer_analysis import (
    compute_transfer_matrix,
    label_schema_similarity,
    vocabulary_overlap,
    detect_negative_transfer,
)
from .error_analysis import categorize_errors, ERROR_CATEGORIES

__all__ = [
    # Metrics
    "compute_task_metrics",
    "bootstrap_ci",
    "paired_permutation_test",
    "wins_ties_losses",
    # Contamination
    "ContaminationChecker",
    "ContaminationResult",
    # Calibration
    "expected_calibration_error",
    "plot_reliability_diagram",
    # Probing
    "LinearProbe",
    "evaluate_adapter_probes",
    # Transfer analysis
    "compute_transfer_matrix",
    "label_schema_similarity",
    "vocabulary_overlap",
    "detect_negative_transfer",
    # Error analysis
    "categorize_errors",
    "ERROR_CATEGORIES",
]
