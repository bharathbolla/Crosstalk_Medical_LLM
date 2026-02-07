"""Projected Conflicting Gradients (PCGrad) optimizer.

Implements gradient projection to reduce negative transfer in multi-task learning.

Reference:
    Gradient Surgery for Multi-Task Learning
    Yu et al., NeurIPS 2020
    https://arxiv.org/abs/2001.06782
"""

from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer


class PCGradOptimizer:
    """PCGrad: Projected Conflicting Gradients for multi-task learning.

    When Task A's gradient conflicts with Task B's gradient (negative cosine
    similarity), project A's gradient onto the normal plane of B's gradient.

    This prevents tasks from interfering with each other during training.

    Used for strategies S3a and S3b only. Adds ~15-20% training overhead.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        model: nn.Module,
        task_names: List[str],
        reduction: str = "mean",
    ):
        """Initialize PCGrad optimizer.

        Args:
            optimizer: Base optimizer (e.g., AdamW)
            model: Model with parameters to optimize
            task_names: List of task names
            reduction: How to combine gradients ("mean" or "sum")
        """
        self.optimizer = optimizer
        self.model = model
        self.task_names = task_names
        self.reduction = reduction

        # Track gradient conflicts for RQ4 analysis
        self.conflict_counts = defaultdict(lambda: defaultdict(int))
        self.total_steps = 0

    def _flatten_gradients(self) -> torch.Tensor:
        """Flatten all parameter gradients into a single vector.

        Returns:
            Flattened gradient tensor
        """
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.view(-1))

        if len(grads) == 0:
            return None

        return torch.cat(grads)

    def _unflatten_gradients(self, flat_grads: torch.Tensor):
        """Unflatten gradient vector back to model parameters.

        Args:
            flat_grads: Flattened gradient tensor
        """
        offset = 0
        for param in self.model.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(flat_grads[offset:offset + numel].view_as(param.grad))
                offset += numel

    def _project_conflicting_gradients(
        self,
        task_gradients: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Project conflicting gradients using PCGrad algorithm.

        Args:
            task_gradients: Dictionary mapping task names to flattened gradients

        Returns:
            Combined projected gradient
        """
        num_tasks = len(task_gradients)
        task_list = list(task_gradients.keys())

        # Convert to list for easier indexing
        grad_list = [task_gradients[task] for task in task_list]

        # PCGrad algorithm
        projected_grads = []

        for i, grad_i in enumerate(grad_list):
            # Start with original gradient
            projected = grad_i.clone()

            # Project onto normal plane of conflicting gradients
            for j, grad_j in enumerate(grad_list):
                if i == j:
                    continue

                # Compute cosine similarity
                dot_product = torch.dot(grad_i, grad_j)
                norm_i = torch.norm(grad_i)
                norm_j = torch.norm(grad_j)

                if norm_i > 0 and norm_j > 0:
                    cos_sim = dot_product / (norm_i * norm_j)

                    # Check for conflict (negative dot product)
                    if dot_product < 0:
                        # Project grad_i onto normal plane of grad_j
                        # proj = grad_i - (grad_i · grad_j / ||grad_j||^2) * grad_j
                        projection_coef = dot_product / (norm_j ** 2)
                        projected = projected - projection_coef * grad_j

                        # Log conflict for RQ4 analysis
                        task_i = task_list[i]
                        task_j = task_list[j]
                        self.conflict_counts[task_i][task_j] += 1

            projected_grads.append(projected)

        # Combine projected gradients
        if self.reduction == "mean":
            combined = torch.stack(projected_grads).mean(dim=0)
        elif self.reduction == "sum":
            combined = torch.stack(projected_grads).sum(dim=0)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        return combined

    def step(self, task_losses: Dict[str, torch.Tensor]):
        """Perform one optimization step with gradient projection.

        Args:
            task_losses: Dictionary mapping task names to loss tensors

        Usage:
            # Compute per-task losses
            task_losses = {}
            for task in tasks:
                loss = model(batch, task)
                task_losses[task] = loss

            # PCGrad step (handles backward and projection)
            pcgrad_optimizer.step(task_losses)
        """
        # Zero gradients
        self.optimizer.zero_grad()

        # Compute gradients for each task separately
        task_gradients = {}

        for task_name, loss in task_losses.items():
            # Backward pass for this task
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # Store flattened gradients
            flat_grad = self._flatten_gradients()
            if flat_grad is not None:
                task_gradients[task_name] = flat_grad.clone()

        # Project conflicting gradients
        if len(task_gradients) > 0:
            combined_grad = self._project_conflicting_gradients(task_gradients)

            # Apply combined gradient to model
            self.optimizer.zero_grad()
            self._unflatten_gradients(combined_grad)

            # Update parameters
            self.optimizer.step()

        # Track total steps
        self.total_steps += 1

    def get_conflict_frequency(self) -> Dict[str, Dict[str, float]]:
        """Get gradient conflict frequency between task pairs.

        Returns:
            Nested dictionary: {task_i: {task_j: frequency}}
            where frequency = conflicts / total_steps

        This is CRITICAL for RQ4 (negative transfer analysis).
        """
        if self.total_steps == 0:
            return {}

        conflict_freq = {}
        for task_i in self.task_names:
            conflict_freq[task_i] = {}
            for task_j in self.task_names:
                if task_i != task_j:
                    count = self.conflict_counts[task_i][task_j]
                    freq = count / self.total_steps
                    conflict_freq[task_i][task_j] = freq

        return conflict_freq

    def get_total_conflicts(self) -> int:
        """Get total number of gradient conflicts observed.

        Returns:
            Total conflict count across all task pairs
        """
        total = 0
        for task_i_conflicts in self.conflict_counts.values():
            total += sum(task_i_conflicts.values())
        return total

    def reset_conflict_stats(self):
        """Reset conflict tracking statistics."""
        self.conflict_counts.clear()
        self.total_steps = 0

    def state_dict(self) -> dict:
        """Get state dict for checkpointing.

        Returns:
            State dictionary with conflict stats
        """
        return {
            "conflict_counts": dict(self.conflict_counts),
            "total_steps": self.total_steps,
            "optimizer_state": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint.

        Args:
            state_dict: State dictionary to load
        """
        self.conflict_counts = defaultdict(
            lambda: defaultdict(int),
            state_dict.get("conflict_counts", {})
        )
        self.total_steps = state_dict.get("total_steps", 0)

        if "optimizer_state" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer_state"])


class SimplePCGrad:
    """Simplified PCGrad for easier integration with HuggingFace Trainer.

    This version modifies gradients in-place during training loop.
    """

    def __init__(self, model: nn.Module, task_names: List[str]):
        """Initialize simple PCGrad.

        Args:
            model: Model with parameters
            task_names: List of task names
        """
        self.model = model
        self.task_names = task_names
        self.conflict_counts = defaultdict(lambda: defaultdict(int))
        self.total_steps = 0

    def project_gradients(self, task_gradients: Dict[str, List[torch.Tensor]]):
        """Project conflicting gradients in-place.

        Args:
            task_gradients: Dictionary mapping task names to list of parameter gradients
        """
        task_list = list(task_gradients.keys())
        num_tasks = len(task_list)

        if num_tasks <= 1:
            return  # No projection needed

        # Flatten gradients
        flat_grads = {}
        for task, grads in task_gradients.items():
            flat = torch.cat([g.view(-1) for g in grads if g is not None])
            flat_grads[task] = flat

        # Project conflicts
        projected = {}
        for i, task_i in enumerate(task_list):
            grad_i = flat_grads[task_i]
            proj_i = grad_i.clone()

            for j, task_j in enumerate(task_list):
                if i == j:
                    continue

                grad_j = flat_grads[task_j]

                # Check for conflict
                dot = torch.dot(grad_i, grad_j)
                if dot < 0:
                    # Project
                    norm_j_sq = torch.dot(grad_j, grad_j)
                    if norm_j_sq > 0:
                        proj_i = proj_i - (dot / norm_j_sq) * grad_j
                        self.conflict_counts[task_i][task_j] += 1

            projected[task_i] = proj_i

        # Average projected gradients
        final_grad = torch.stack(list(projected.values())).mean(dim=0)

        # Unflatten and assign to model parameters
        offset = 0
        for param in self.model.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(final_grad[offset:offset + numel].view_as(param.grad))
                offset += numel

        self.total_steps += 1

    def get_conflict_frequency(self) -> Dict[str, Dict[str, float]]:
        """Get conflict frequency matrix."""
        if self.total_steps == 0:
            return {}

        freq = {}
        for task_i in self.task_names:
            freq[task_i] = {}
            for task_j in self.task_names:
                if task_i != task_j:
                    count = self.conflict_counts[task_i][task_j]
                    freq[task_i][task_j] = count / self.total_steps

        return freq
