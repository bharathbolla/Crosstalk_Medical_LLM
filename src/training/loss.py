"""Multi-task loss functions with learned task weighting.

Implements uncertainty-weighted loss (Kendall et al., 2018) and equal weighting baseline.
"""

from typing import Dict, List

import torch
import torch.nn as nn


class UncertaintyWeightedLoss(nn.Module):
    """Learns task weights via homoscedastic uncertainty (Kendall et al., 2018).

    Loss formula:
        L = sum_i (1 / (2 * sigma_i^2)) * L_i + log(sigma_i)

    where sigma_i is a learned uncertainty parameter for task i.

    The model learns to up-weight easy tasks (low uncertainty) and
    down-weight difficult tasks (high uncertainty), while the log term
    prevents collapse to sigma → infinity.

    Reference:
        Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
        Kendall et al., CVPR 2018
    """

    def __init__(self, task_names: List[str], init_log_sigma: float = 0.0):
        """Initialize uncertainty-weighted loss.

        Args:
            task_names: List of task names
            init_log_sigma: Initial value for log(sigma) (default: 0.0 → sigma=1.0)
        """
        super().__init__()
        self.task_names = task_names
        self.num_tasks = len(task_names)

        # Learned log(sigma) for numerical stability
        # We learn log(sigma) instead of sigma to ensure positivity
        self.log_sigmas = nn.Parameter(
            torch.ones(self.num_tasks) * init_log_sigma
        )

        # Create task name to index mapping
        self.task_to_idx = {name: idx for idx, name in enumerate(task_names)}

    def forward(
        self,
        task_losses: Dict[str, torch.Tensor],
        return_weights: bool = False,
    ) -> torch.Tensor:
        """Compute weighted multi-task loss.

        Args:
            task_losses: Dictionary mapping task names to loss tensors
            return_weights: Whether to return task weights (for logging)

        Returns:
            Weighted total loss, optionally with task weights dict
        """
        total_loss = 0.0
        task_weights = {}

        for task_name, loss in task_losses.items():
            if task_name not in self.task_to_idx:
                raise ValueError(f"Unknown task: {task_name}")

            idx = self.task_to_idx[task_name]
            log_sigma = self.log_sigmas[idx]

            # Uncertainty-weighted term: (1 / (2 * sigma^2)) * L_i
            # Using log_sigma: = (1 / (2 * exp(2 * log_sigma))) * L_i
            #                  = exp(-2 * log_sigma) / 2 * L_i
            precision = torch.exp(-2 * log_sigma)
            weighted_loss = 0.5 * precision * loss

            # Regularization term: log(sigma)
            # Prevents sigma from going to infinity
            reg_term = log_sigma

            task_total = weighted_loss + reg_term
            total_loss = total_loss + task_total

            # Store weight for logging
            if return_weights:
                task_weights[task_name] = precision.item()

        if return_weights:
            return total_loss, task_weights

        return total_loss

    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights (1 / sigma^2).

        Returns:
            Dictionary mapping task names to weights
        """
        weights = {}
        with torch.no_grad():
            for task_name, idx in self.task_to_idx.items():
                precision = torch.exp(-2 * self.log_sigmas[idx])
                weights[task_name] = precision.item()

        return weights

    def get_task_uncertainties(self) -> Dict[str, float]:
        """Get current task uncertainties (sigma).

        Returns:
            Dictionary mapping task names to uncertainty values
        """
        uncertainties = {}
        with torch.no_grad():
            for task_name, idx in self.task_to_idx.items():
                sigma = torch.exp(self.log_sigmas[idx])
                uncertainties[task_name] = sigma.item()

        return uncertainties


class EqualWeightedLoss(nn.Module):
    """Baseline: simple average of task losses.

    Loss formula:
        L = sum_i L_i / num_tasks

    This serves as the ablation baseline to test whether learned
    uncertainty weighting provides benefits.
    """

    def __init__(self, task_names: List[str]):
        """Initialize equal-weighted loss.

        Args:
            task_names: List of task names
        """
        super().__init__()
        self.task_names = task_names
        self.num_tasks = len(task_names)

    def forward(
        self,
        task_losses: Dict[str, torch.Tensor],
        return_weights: bool = False,
    ) -> torch.Tensor:
        """Compute equal-weighted multi-task loss.

        Args:
            task_losses: Dictionary mapping task names to loss tensors
            return_weights: Whether to return task weights (for logging)

        Returns:
            Average loss, optionally with uniform weights dict
        """
        # Simple average
        total_loss = sum(task_losses.values()) / len(task_losses)

        if return_weights:
            # All tasks have equal weight
            task_weights = {task: 1.0 / len(task_losses) for task in task_losses.keys()}
            return total_loss, task_weights

        return total_loss

    def get_task_weights(self) -> Dict[str, float]:
        """Get task weights (all equal).

        Returns:
            Dictionary mapping task names to uniform weights
        """
        return {task: 1.0 / self.num_tasks for task in self.task_names}


class ManualWeightedLoss(nn.Module):
    """Fixed manual task weights (alternative to learned weighting).

    Useful for:
    - Balancing task difficulties
    - Prioritizing specific tasks
    - Ablation studies
    """

    def __init__(self, task_weights: Dict[str, float]):
        """Initialize manual-weighted loss.

        Args:
            task_weights: Dictionary mapping task names to fixed weights
        """
        super().__init__()
        self.task_weights = task_weights

        # Normalize weights to sum to 1
        total_weight = sum(task_weights.values())
        self.task_weights = {
            task: weight / total_weight
            for task, weight in task_weights.items()
        }

    def forward(
        self,
        task_losses: Dict[str, torch.Tensor],
        return_weights: bool = False,
    ) -> torch.Tensor:
        """Compute manually-weighted multi-task loss.

        Args:
            task_losses: Dictionary mapping task names to loss tensors
            return_weights: Whether to return task weights (for logging)

        Returns:
            Weighted loss, optionally with weights dict
        """
        total_loss = 0.0

        for task_name, loss in task_losses.items():
            if task_name not in self.task_weights:
                raise ValueError(f"No weight specified for task: {task_name}")

            weight = self.task_weights[task_name]
            total_loss = total_loss + weight * loss

        if return_weights:
            return total_loss, self.task_weights.copy()

        return total_loss

    def get_task_weights(self) -> Dict[str, float]:
        """Get task weights.

        Returns:
            Dictionary of fixed task weights
        """
        return self.task_weights.copy()
