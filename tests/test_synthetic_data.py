"""Unit tests using synthetic data.

Tests core functionality without requiring:
- Real medical datasets (PhysioNet access)
- Model weights
- GPU resources

Based on TRAINING_EVALUATION_SUMMARY.md Phase 0 testing strategy.
"""

import sys
from pathlib import Path
import unittest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTokenTracker(unittest.TestCase):
    """Test TokenTracker with synthetic data (RQ5 critical)."""

    def setUp(self):
        """Import modules (skip if deps missing)."""
        try:
            from src.data.multitask_loader import TokenTracker
            self.TokenTracker = TokenTracker
            self.has_deps = True
        except ImportError:
            self.has_deps = False

    def test_token_tracking(self):
        """Test token counting across tasks."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        tracker = self.TokenTracker()

        # Simulate training
        tracker.update("task1", 100, step=0)
        tracker.update("task2", 150, step=1)
        tracker.update("task1", 200, step=2)

        # Check totals
        self.assertEqual(tracker.get_total_tokens(), 450)
        self.assertEqual(tracker.tokens_per_task["task1"], 300)
        self.assertEqual(tracker.tokens_per_task["task2"], 150)

    def test_token_distribution(self):
        """Test percentage distribution calculation."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        tracker = self.TokenTracker()
        tracker.update("task1", 600)
        tracker.update("task2", 300)
        tracker.update("task3", 100)

        dist = tracker.get_token_distribution()
        self.assertAlmostEqual(dist["task1"], 0.6)
        self.assertAlmostEqual(dist["task2"], 0.3)
        self.assertAlmostEqual(dist["task3"], 0.1)

    def test_checkpoint_save_load(self):
        """Test state save/restore."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        tracker = self.TokenTracker()
        tracker.update("task1", 500)
        tracker.update("task2", 300)

        # Save state
        state = tracker.get_state()

        # Create new tracker and restore
        new_tracker = self.TokenTracker()
        new_tracker.load_state(state)

        self.assertEqual(new_tracker.get_total_tokens(), 800)
        self.assertEqual(new_tracker.tokens_per_task["task1"], 500)


class TestStatisticalFunctions(unittest.TestCase):
    """Test bootstrap CI and permutation tests with random data."""

    def setUp(self):
        """Import modules."""
        try:
            import numpy as np
            from src.evaluation.metrics import (
                bootstrap_ci,
                paired_permutation_test,
                wins_ties_losses,
            )
            self.np = np
            self.bootstrap_ci = bootstrap_ci
            self.paired_permutation_test = paired_permutation_test
            self.wins_ties_losses = wins_ties_losses
            self.has_deps = True
        except ImportError:
            self.has_deps = False

    def test_bootstrap_ci(self):
        """Test bootstrap confidence interval."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        # Random scores
        scores = self.np.random.randn(100)

        # Compute CI
        lower, upper = self.bootstrap_ci(scores, n_bootstrap=1000)

        # Sanity checks
        self.assertLess(lower, upper)
        self.assertGreater(upper - lower, 0)

    def test_permutation_test(self):
        """Test paired permutation test."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        # Identical distributions (should give high p-value)
        scores_a = self.np.random.randn(100)
        scores_b = scores_a + self.np.random.randn(100) * 0.1

        p_value = self.paired_permutation_test(
            scores_a, scores_b, n_permutations=1000
        )

        # p-value should be in [0, 1]
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_wins_ties_losses(self):
        """Test W/T/L counting."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        scores_a = self.np.array([0.8, 0.6, 0.5, 0.7])
        scores_b = self.np.array([0.7, 0.6, 0.6, 0.8])

        wins, ties, losses = self.wins_ties_losses(scores_a, scores_b, threshold=0.05)

        # Check counts
        self.assertEqual(wins + ties + losses, 4)
        self.assertGreaterEqual(wins, 0)
        self.assertGreaterEqual(ties, 0)
        self.assertGreaterEqual(losses, 0)


class TestCalibration(unittest.TestCase):
    """Test ECE computation with random predictions."""

    def setUp(self):
        """Import modules."""
        try:
            import numpy as np
            from src.evaluation.calibration import expected_calibration_error
            self.np = np
            self.expected_calibration_error = expected_calibration_error
            self.has_deps = True
        except ImportError:
            self.has_deps = False

    def test_perfect_calibration(self):
        """Test ECE with perfectly calibrated predictions."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        # Generate perfectly calibrated data
        confidences = self.np.random.rand(1000)
        predictions = (self.np.random.rand(1000) < confidences).astype(int)
        labels = predictions  # Perfect accuracy

        ece, _, _, _ = self.expected_calibration_error(
            confidences, predictions, labels, n_bins=10
        )

        # ECE should be very low for perfect calibration
        self.assertLess(ece, 0.1)

    def test_random_calibration(self):
        """Test ECE with random predictions."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        confidences = self.np.random.rand(1000)
        predictions = self.np.random.randint(0, 2, 1000)
        labels = self.np.random.randint(0, 2, 1000)

        ece, bin_accs, bin_confs, bin_counts = self.expected_calibration_error(
            confidences, predictions, labels, n_bins=15
        )

        # ECE should be in valid range
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

        # Bins should sum to total samples
        self.assertEqual(sum(bin_counts), 1000)


class TestTransferAnalysis(unittest.TestCase):
    """Test transfer analysis functions with synthetic scores."""

    def setUp(self):
        """Import modules."""
        try:
            import numpy as np
            from src.evaluation.transfer_analysis import (
                detect_negative_transfer,
                predict_transfer_success,
                label_schema_similarity,
            )
            self.np = np
            self.detect_negative_transfer = detect_negative_transfer
            self.predict_transfer_success = predict_transfer_success
            self.label_schema_similarity = label_schema_similarity
            self.has_deps = True
        except ImportError:
            self.has_deps = False

    def test_negative_transfer_detection(self):
        """Test negative transfer detection."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        single_task = {"task1": 0.8, "task2": 0.7, "task3": 0.6}
        multi_task = {"task1": 0.85, "task2": 0.65, "task3": 0.55}

        # task2 and task3 show negative transfer (>0.5 drop with margin=0.5)
        negative_tasks = self.detect_negative_transfer(
            single_task, multi_task, noise_margin=0.03
        )

        self.assertIn("task2", negative_tasks)
        self.assertIn("task3", negative_tasks)
        self.assertNotIn("task1", negative_tasks)

    def test_transfer_correlation(self):
        """Test similarity-transfer correlation."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        # Random matrices
        similarity_matrix = self.np.random.rand(5, 5)
        transfer_matrix = self.np.random.randn(5, 5)

        rho, p_value = self.predict_transfer_success(
            similarity_matrix, transfer_matrix
        )

        # Check valid correlation coefficient
        self.assertGreaterEqual(rho, -1.0)
        self.assertLessEqual(rho, 1.0)
        self.assertGreaterEqual(p_value, 0.0)
        self.assertLessEqual(p_value, 1.0)

    def test_label_similarity(self):
        """Test Jaccard similarity of label sets."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        task_configs = {
            "task_a": {"labels": ["Drug", "Symptom", "Disorder"]},
            "task_b": {"labels": ["Symptom", "Disorder", "Procedure"]},
        }

        similarity = self.label_schema_similarity("task_a", "task_b", task_configs)

        # Intersection: {Symptom, Disorder} = 2
        # Union: {Drug, Symptom, Disorder, Procedure} = 4
        # Jaccard: 2/4 = 0.5
        self.assertAlmostEqual(similarity, 0.5)


class TestLossFunctions(unittest.TestCase):
    """Test multi-task loss functions with synthetic losses."""

    def setUp(self):
        """Import modules."""
        try:
            import torch
            from src.training.loss import UncertaintyWeightedLoss, EqualWeightedLoss
            self.torch = torch
            self.UncertaintyWeightedLoss = UncertaintyWeightedLoss
            self.EqualWeightedLoss = EqualWeightedLoss
            self.has_deps = True
        except ImportError:
            self.has_deps = False

    def test_uncertainty_weighted_loss(self):
        """Test uncertainty-weighted loss computation."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        loss_fn = self.UncertaintyWeightedLoss(["task1", "task2", "task3"])

        task_losses = {
            "task1": self.torch.tensor(0.5),
            "task2": self.torch.tensor(0.8),
            "task3": self.torch.tensor(0.3),
        }

        total_loss = loss_fn(task_losses)

        # Check loss is positive
        self.assertGreater(total_loss.item(), 0)

        # Check weights are learnable
        self.assertTrue(loss_fn.log_sigmas.requires_grad)

    def test_equal_weighted_loss(self):
        """Test simple average loss."""
        if not self.has_deps:
            self.skipTest("Dependencies not installed")

        loss_fn = self.EqualWeightedLoss(["task1", "task2"])

        task_losses = {
            "task1": self.torch.tensor(0.6),
            "task2": self.torch.tensor(0.4),
        }

        total_loss = loss_fn(task_losses)

        # Should be simple average: (0.6 + 0.4) / 2 = 0.5
        self.assertAlmostEqual(total_loss.item(), 0.5, places=5)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
