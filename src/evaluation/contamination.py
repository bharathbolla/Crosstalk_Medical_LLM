"""Contamination detection with three-layer protocol.

Phase 0: Run before any training to check if test data was in pretraining.

Three layers:
1. Zero-shot audit: Run base model (no fine-tuning) on test set
2. N-gram overlap: Check for verbatim text overlap
3. Min-K% probing: Statistical test on log-likelihoods
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats

import torch
import torch.nn as nn


@dataclass
class ContaminationResult:
    """Result of contamination check for a model-task pair."""
    model_name: str
    task_name: str

    # Layer 1: Zero-shot
    zero_shot_f1: float
    zero_shot_threshold: float  # 70% of published SOTA
    zero_shot_contaminated: bool

    # Layer 2: N-gram overlap
    ngram_overlap: Dict[int, float]  # n → overlap ratio
    ngram_threshold: float  # 5% samples with 13-gram match
    ngram_contaminated: bool

    # Layer 3: Min-K% probing
    min_k_p_value: float
    min_k_threshold: float  # p < 0.01
    min_k_contaminated: bool

    # Overall
    overall_contaminated: bool

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "task_name": self.task_name,
            "zero_shot_f1": self.zero_shot_f1,
            "zero_shot_contaminated": self.zero_shot_contaminated,
            "ngram_overlap": self.ngram_overlap,
            "ngram_contaminated": self.ngram_contaminated,
            "min_k_p_value": self.min_k_p_value,
            "min_k_contaminated": self.min_k_contaminated,
            "overall_contaminated": self.overall_contaminated,
        }


class ContaminationChecker:
    """Three-layer contamination detection."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        task_name: str,
        sota_score: float,
    ):
        """Initialize contamination checker.

        Args:
            model: Base model (no fine-tuning)
            tokenizer: Tokenizer
            task_name: Task to check
            sota_score: Published SOTA score for this task
        """
        self.model = model
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.sota_score = sota_score

    def check(
        self,
        test_data,
        control_data: Optional[Any] = None,
    ) -> ContaminationResult:
        """Run all three contamination checks.

        Args:
            test_data: Test dataset
            control_data: Control clinical text (for Min-K%)

        Returns:
            ContaminationResult
        """
        # Layer 1: Zero-shot audit
        zero_shot_f1 = self._zero_shot_audit(test_data)
        zero_shot_threshold = 0.7 * self.sota_score
        zero_shot_contaminated = zero_shot_f1 > zero_shot_threshold

        # Layer 2: N-gram overlap
        ngram_overlap = self._ngram_overlap(test_data)
        ngram_contaminated = ngram_overlap.get(13, 0) > 0.05

        # Layer 3: Min-K% probing
        min_k_p_value = self._min_k_probing(test_data, control_data)
        min_k_contaminated = min_k_p_value < 0.01

        # Overall contamination
        overall_contaminated = (
            zero_shot_contaminated or
            ngram_contaminated or
            min_k_contaminated
        )

        return ContaminationResult(
            model_name=self.model.config.name_or_path,
            task_name=self.task_name,
            zero_shot_f1=zero_shot_f1,
            zero_shot_threshold=zero_shot_threshold,
            zero_shot_contaminated=zero_shot_contaminated,
            ngram_overlap=ngram_overlap,
            ngram_threshold=0.05,
            ngram_contaminated=ngram_contaminated,
            min_k_p_value=min_k_p_value,
            min_k_threshold=0.01,
            min_k_contaminated=min_k_contaminated,
            overall_contaminated=overall_contaminated,
        )

    def _zero_shot_audit(self, test_data) -> float:
        """Layer 1: Run base model on test set.

        Returns:
            Zero-shot F1 score
        """
        # TODO: Implement zero-shot evaluation
        # Requires actual model inference on test data
        return 0.0

    def _ngram_overlap(
        self,
        test_data,
        n_values: List[int] = [8, 10, 13],
    ) -> Dict[int, float]:
        """Layer 2: Check for verbatim n-gram matches.

        Args:
            test_data: Test dataset
            n_values: N-gram sizes to check

        Returns:
            Dict mapping n → overlap ratio
        """
        # TODO: Implement n-gram overlap check
        # Requires generating continuations and checking for matches
        return {n: 0.0 for n in n_values}

    def _min_k_probing(
        self,
        test_data,
        control_data,
        k_percent: float = 20.0,
    ) -> float:
        """Layer 3: Min-K% statistical test.

        Compares log-likelihoods of test vs control data.

        Args:
            test_data: Test dataset
            control_data: Control clinical text
            k_percent: Percentage of lowest-likelihood tokens to consider

        Returns:
            P-value from Mann-Whitney U test
        """
        # TODO: Implement Min-K% probing
        # Requires computing log-likelihoods for both datasets
        return 1.0  # Placeholder: no contamination detected
