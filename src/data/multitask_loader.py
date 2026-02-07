"""Multi-task batch sampling with token tracking.

Critical for RQ5: TokenTracker logs cumulative tokens per task, enabling
the token-controlled baseline comparison.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .base import BaseTaskDataset, UnifiedSample


class TokenTracker:
    """Tracks cumulative tokens processed per task across training.

    This is the CRITICAL component for RQ5: by logging tokens_per_task at every
    checkpoint, we enable the token-controlled baseline to match the exact data
    exposure of multi-task runs.

    Design principles:
    - Tracks tokens per task, not total tokens (we need per-task breakdown)
    - Updates every batch (accumulated during collation)
    - Saves to checkpoint (must persist across Kaggle session restarts)
    - Provides query interface for TokenControlledTrainer
    """

    def __init__(self, tasks: List[str], checkpoint_path: Optional[Path] = None):
        """Initialize token tracker.

        Args:
            tasks: List of task names to track
            checkpoint_path: Path to load existing token counts (for resume)
        """
        self.tasks = tasks
        self.tokens_per_task: Dict[str, int] = {task: 0 for task in tasks}
        self.tokens_per_step: Dict[int, Dict[str, int]] = {}  # For detailed logging
        self.current_step = 0

        # Load from checkpoint if provided
        if checkpoint_path and checkpoint_path.exists():
            self.load(checkpoint_path)

    def update(self, task: str, token_count: int, step: Optional[int] = None):
        """Update token count for a task.

        Called by collator after tokenization of each batch.

        Args:
            task: Task name
            token_count: Number of tokens in the batch
            step: Training step (optional, for detailed logging)
        """
        if task not in self.tokens_per_task:
            raise ValueError(f"Unknown task: {task}. Registered tasks: {self.tasks}")

        self.tokens_per_task[task] += token_count

        # Log per-step breakdown for analysis
        if step is not None:
            self.current_step = step
            if step not in self.tokens_per_step:
                self.tokens_per_step[step] = {task: 0 for task in self.tasks}
            self.tokens_per_step[step][task] += token_count

    def get_total_tokens(self, task: Optional[str] = None) -> int:
        """Get total tokens processed.

        Args:
            task: If provided, return tokens for that task only.
                  If None, return sum across all tasks.

        Returns:
            Total token count
        """
        if task is not None:
            return self.tokens_per_task.get(task, 0)
        return sum(self.tokens_per_task.values())

    def get_token_distribution(self) -> Dict[str, float]:
        """Get percentage distribution of tokens across tasks.

        Returns:
            Dictionary mapping task to percentage of total tokens
        """
        total = self.get_total_tokens()
        if total == 0:
            return {task: 0.0 for task in self.tasks}
        return {task: count / total for task, count in self.tokens_per_task.items()}

    def save(self, path: Path):
        """Save token counts to JSON file.

        Args:
            path: File path to save to
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tokens_per_task": self.tokens_per_task,
            "total_tokens": self.get_total_tokens(),
            "current_step": self.current_step,
            "distribution": self.get_token_distribution(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path):
        """Load token counts from JSON file.

        Args:
            path: File path to load from
        """
        with open(path, "r") as f:
            data = json.load(f)
        self.tokens_per_task = data["tokens_per_task"]
        self.current_step = data.get("current_step", 0)

    def __repr__(self) -> str:
        total = self.get_total_tokens()
        dist = self.get_token_distribution()
        lines = [f"TokenTracker(total={total:,})"]
        for task, count in sorted(self.tokens_per_task.items()):
            pct = dist[task] * 100
            lines.append(f"  {task}: {count:,} ({pct:.1f}%)")
        return "\n".join(lines)


SamplingStrategy = Literal["proportional", "uniform", "temperature"]


class MultiTaskBatchSampler(Sampler):
    """Batch sampler that handles multiple task datasets simultaneously.

    Supports three sampling strategies:
    1. Proportional: Sample tasks based on dataset sizes
    2. Uniform: Sample each task with equal probability
    3. Temperature-scaled: Smooth between proportional and uniform

    Design principles:
    - Works with DataLoader to provide (task_idx, batch_indices) tuples
    - Supports resuming from a specific step (critical for Kaggle)
    - Ensures each task gets at least min_samples_per_task per epoch
    - Deterministic given a seed (for reproducibility)
    """

    def __init__(
        self,
        datasets: Dict[str, BaseTaskDataset],
        batch_size: int,
        strategy: SamplingStrategy = "proportional",
        temperature: float = 1.0,
        seed: int = 42,
        drop_last: bool = True,
        min_samples_per_task: Optional[int] = None,
        start_step: int = 0,
    ):
        """Initialize multi-task batch sampler.

        Args:
            datasets: Dictionary mapping task names to datasets
            batch_size: Batch size per task
            strategy: Sampling strategy (proportional, uniform, temperature)
            temperature: Temperature parameter for temperature-scaled sampling
                        τ=1.0 → proportional, τ=0.0 → uniform, τ>1.0 → more skewed
            seed: Random seed for reproducibility
            drop_last: Whether to drop incomplete batches
            min_samples_per_task: Minimum samples per task per epoch
            start_step: Starting step (for resuming from checkpoint)
        """
        self.datasets = datasets
        self.task_names = list(datasets.keys())
        self.batch_size = batch_size
        self.strategy = strategy
        self.temperature = temperature
        self.seed = seed
        self.drop_last = drop_last
        self.start_step = start_step

        # Calculate dataset sizes and sampling weights
        self.dataset_sizes = {task: len(ds) for task, ds in datasets.items()}
        self.total_samples = sum(self.dataset_sizes.values())

        # Calculate number of batches per task
        self.batches_per_task = {
            task: size // batch_size if drop_last else math.ceil(size / batch_size)
            for task, size in self.dataset_sizes.items()
        }
        self.total_batches = sum(self.batches_per_task.values())

        # Set minimum samples per task per epoch
        if min_samples_per_task is None:
            # Default: ensure each task gets at least 5% of total batches
            min_batches = max(1, int(0.05 * self.total_batches))
            self.min_samples_per_task = min_batches * batch_size
        else:
            self.min_samples_per_task = min_samples_per_task

        # Calculate sampling probabilities
        self.sampling_probs = self._compute_sampling_probs()

        # Initialize RNG
        self.rng = np.random.RandomState(seed)

    def _compute_sampling_probs(self) -> Dict[str, float]:
        """Compute sampling probabilities for each task.

        Returns:
            Dictionary mapping task names to sampling probabilities
        """
        if self.strategy == "uniform":
            # Equal probability for all tasks
            prob = 1.0 / len(self.task_names)
            return {task: prob for task in self.task_names}

        elif self.strategy == "proportional":
            # Probability proportional to dataset size
            return {
                task: size / self.total_samples
                for task, size in self.dataset_sizes.items()
            }

        elif self.strategy == "temperature":
            # Temperature-scaled smoothing
            # p_i ∝ (n_i)^(1/τ) where n_i is dataset size
            if self.temperature <= 0:
                raise ValueError("Temperature must be positive")

            # Apply temperature scaling
            scaled_sizes = {
                task: size ** (1.0 / self.temperature)
                for task, size in self.dataset_sizes.items()
            }
            total_scaled = sum(scaled_sizes.values())

            return {task: scaled / total_scaled for task, scaled in scaled_sizes.items()}

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def __iter__(self) -> Iterator[tuple]:
        """Iterate over batches.

        Yields:
            (task_name, batch_indices) tuples
        """
        # Reset RNG for reproducibility
        self.rng = np.random.RandomState(self.seed)

        # Create per-task index pools (shuffled)
        task_indices = {}
        for task in self.task_names:
            indices = list(range(self.dataset_sizes[task]))
            self.rng.shuffle(indices)
            task_indices[task] = indices

        # Track samples yielded per task
        samples_yielded = {task: 0 for task in self.task_names}

        step = 0
        while True:
            # Check if we've exhausted all tasks
            available_tasks = [
                task for task in self.task_names
                if len(task_indices[task]) >= self.batch_size
            ]

            if not available_tasks:
                # Check if minimum samples requirement is met
                insufficient = [
                    task for task in self.task_names
                    if samples_yielded[task] < self.min_samples_per_task
                ]
                if insufficient:
                    # Reshuffle and continue for insufficient tasks
                    for task in insufficient:
                        indices = list(range(self.dataset_sizes[task]))
                        self.rng.shuffle(indices)
                        task_indices[task] = indices
                    continue
                else:
                    # All tasks meet minimum requirement, stop
                    break

            # Sample a task based on strategy
            task_probs = [self.sampling_probs[task] for task in available_tasks]
            task_probs = np.array(task_probs) / sum(task_probs)  # Renormalize
            task = self.rng.choice(available_tasks, p=task_probs)

            # Extract batch
            batch_indices = task_indices[task][:self.batch_size]
            task_indices[task] = task_indices[task][self.batch_size:]

            # Skip if we're resuming from a checkpoint
            if step < self.start_step:
                step += 1
                continue

            samples_yielded[task] += len(batch_indices)
            step += 1

            yield (task, batch_indices)

    def __len__(self) -> int:
        """Return approximate number of batches per epoch."""
        return self.total_batches


def create_multitask_dataloader(
    datasets: Dict[str, BaseTaskDataset],
    batch_size: int,
    collate_fn,
    strategy: SamplingStrategy = "proportional",
    temperature: float = 1.0,
    num_workers: int = 0,
    seed: int = 42,
    start_step: int = 0,
) -> DataLoader:
    """Create a DataLoader with multi-task batch sampling.

    Args:
        datasets: Dictionary mapping task names to datasets
        batch_size: Batch size per task
        collate_fn: Collator that accepts (task_name, samples) and returns batch dict
        strategy: Sampling strategy
        temperature: Temperature for temperature-scaled sampling
        num_workers: Number of data loading workers
        seed: Random seed
        start_step: Starting step for resuming

    Returns:
        DataLoader configured for multi-task training
    """
    sampler = MultiTaskBatchSampler(
        datasets=datasets,
        batch_size=batch_size,
        strategy=strategy,
        temperature=temperature,
        seed=seed,
        start_step=start_step,
    )

    # Custom collate function that handles (task, indices) tuples
    def multitask_collate(batch_info):
        task, indices = batch_info[0]  # DataLoader wraps in a list
        dataset = datasets[task]
        samples = [dataset[i] for i in indices]
        return collate_fn(task, samples)

    return DataLoader(
        dataset=list(datasets.values())[0],  # Dummy dataset, sampler controls iteration
        batch_sampler=sampler,
        collate_fn=multitask_collate,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
