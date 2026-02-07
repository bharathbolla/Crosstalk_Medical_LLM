"""Calibration metrics and reliability diagrams.

Expected Calibration Error (ECE) measures if model confidence matches accuracy.
Critical for medical applications: models must know when they're uncertain.
"""

from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt


def expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, List[float], List[float], List[int]]:
    """Compute Expected Calibration Error (ECE).

    ECE measures calibration: if model says 90% confident,
    it should be correct 90% of the time.

    Args:
        confidences: Model confidence scores [0, 1]
        predictions: Predicted labels
        labels: True labels
        n_bins: Number of bins for calibration

    Returns:
        Tuple of (ece, bin_accuracies, bin_confidences, bin_counts)
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Initialize bin statistics
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    # Compute per-bin statistics
    ece = 0.0
    total_samples = len(predictions)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        n_in_bin = in_bin.sum()

        if n_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()

            # Average confidence in this bin
            avg_confidence_in_bin = confidences[in_bin].mean()

            # Contribution to ECE
            ece += (n_in_bin / total_samples) * abs(accuracy_in_bin - avg_confidence_in_bin)

            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(n_in_bin)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    return ece, bin_accuracies, bin_confidences, bin_counts


def plot_reliability_diagram(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    save_path: Optional[str] = None,
):
    """Plot reliability diagram (calibration plot).

    Args:
        confidences: Model confidence scores
        predictions: Predicted labels
        labels: True labels
        n_bins: Number of bins
        save_path: Path to save figure (optional)
    """
    ece, bin_accuracies, bin_confidences, bin_counts = expected_calibration_error(
        confidences, predictions, labels, n_bins
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    # Plot model calibration
    ax.bar(
        bin_confidences,
        bin_accuracies,
        width=1.0 / n_bins,
        alpha=0.7,
        label=f'Model (ECE={ece:.3f})'
    )

    ax.set_xlabel('Confidence', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Reliability Diagram', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

    return ece
