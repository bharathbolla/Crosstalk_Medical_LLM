"""Multi-task trainer with token tracking and PCGrad support.

Extends HuggingFace Trainer for multi-task learning with:
- Token counting for RQ5
- PCGrad for S3a/S3b
- Kaggle-safe checkpointing
- VRAM monitoring
"""

from typing import Dict, Optional, List, Callable, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from ..data import TokenTracker, MultiTaskBatchSampler
from ..utils import VRAMMonitor, CheckpointManager, smoke_test
from ..models import MultiTaskModel

from .loss import UncertaintyWeightedLoss, EqualWeightedLoss
from .pcgrad import PCGradOptimizer
from .callbacks import (
    VRAMCallback,
    QuickEvalCallback,
    TokenLoggingCallback,
    GradientConflictCallback,
)


class MultiTaskTrainer(Trainer):
    """Multi-task trainer with token tracking and gradient conflict resolution.

    Key features:
    - Dynamic task sampling (proportional/uniform/temperature)
    - Per-task loss tracking
    - Token counting for RQ5
    - PCGrad integration for S3a/S3b
    - Checkpoint resume for Kaggle
    """

    def __init__(
        self,
        model: MultiTaskModel,
        args: TrainingArguments,
        train_datasets: Dict[str, Any],
        eval_datasets: Dict[str, Any],
        task_configs: Dict[str, Dict],
        strategy: str,
        token_tracker: Optional[TokenTracker] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        vram_monitor: Optional[VRAMMonitor] = None,
        use_pcgrad: bool = False,
        use_uncertainty_weighting: bool = True,
        sampling_strategy: str = "proportional",
        sampling_temperature: float = 1.0,
        **kwargs,
    ):
        """Initialize multi-task trainer.

        Args:
            model: MultiTaskModel instance
            args: Training arguments
            train_datasets: Dictionary of training datasets by task
            eval_datasets: Dictionary of evaluation datasets by task
            task_configs: Task configurations
            strategy: Training strategy (S1, S2, S3a, S3b, S4, S5)
            token_tracker: TokenTracker for RQ5 (optional)
            checkpoint_manager: CheckpointManager for Kaggle safety (optional)
            vram_monitor: VRAMMonitor for leak detection (optional)
            use_pcgrad: Whether to use PCGrad (for S3a/S3b)
            use_uncertainty_weighting: Whether to use uncertainty-weighted loss
            sampling_strategy: Task sampling strategy
            sampling_temperature: Temperature for temperature-scaled sampling
            **kwargs: Additional arguments for Trainer
        """
        # Initialize HuggingFace Trainer
        super().__init__(
            model=model,
            args=args,
            **kwargs
        )

        self.train_datasets = train_datasets
        self.eval_datasets = eval_datasets
        self.task_configs = task_configs
        self.strategy = strategy
        self.sampling_strategy = sampling_strategy
        self.sampling_temperature = sampling_temperature

        # Token tracking (CRITICAL for RQ5)
        if token_tracker is None:
            self.token_tracker = TokenTracker(list(train_datasets.keys()))
        else:
            self.token_tracker = token_tracker

        # Checkpoint management (Kaggle safety)
        if checkpoint_manager is None:
            checkpoint_dir = Path(args.output_dir) / "checkpoints"
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        else:
            self.checkpoint_manager = checkpoint_manager

        # VRAM monitoring
        if vram_monitor is None:
            self.vram_monitor = VRAMMonitor(check_interval=50)
        else:
            self.vram_monitor = vram_monitor

        # Multi-task loss
        task_names = list(train_datasets.keys())
        if use_uncertainty_weighting:
            self.multi_task_loss = UncertaintyWeightedLoss(task_names)
        else:
            self.multi_task_loss = EqualWeightedLoss(task_names)

        # PCGrad optimizer (for S3a/S3b)
        self.use_pcgrad = use_pcgrad
        self.pcgrad_optimizer = None
        if use_pcgrad and self.optimizer is not None:
            self.pcgrad_optimizer = PCGradOptimizer(
                optimizer=self.optimizer,
                model=model,
                task_names=task_names,
            )

        # Add callbacks
        self._add_custom_callbacks()

    def _add_custom_callbacks(self):
        """Add custom callbacks for monitoring."""
        # VRAM monitoring
        self.add_callback(VRAMCallback(check_interval=50))

        # Token logging (RQ5)
        self.add_callback(TokenLoggingCallback(self.token_tracker))

        # Gradient conflicts (RQ4, if using PCGrad)
        if self.pcgrad_optimizer is not None:
            self.add_callback(GradientConflictCallback(self.pcgrad_optimizer))

    def get_train_dataloader(self) -> DataLoader:
        """Create multi-task training dataloader.

        Returns:
            DataLoader with multi-task batch sampling
        """
        from ..data import create_multitask_dataloader

        # Get appropriate collator
        # TODO: This should be task-aware and use the right collator per task
        collate_fn = self.data_collator

        dataloader = create_multitask_dataloader(
            datasets=self.train_datasets,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=collate_fn,
            strategy=self.sampling_strategy,
            temperature=self.sampling_temperature,
            num_workers=self.args.dataloader_num_workers,
            seed=self.args.seed,
        )

        return dataloader

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ):
        """Compute multi-task loss.

        Args:
            model: Model
            inputs: Batch inputs (includes 'task' key)
            return_outputs: Whether to return outputs

        Returns:
            Loss, optionally with outputs
        """
        task = inputs.pop("task", None)

        if task is None:
            raise ValueError("Batch must include 'task' key for multi-task training")

        # Forward pass
        outputs = model(**inputs, task=task)

        # Extract loss
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        else:
            raise ValueError("Model must return loss in outputs dict")

        # Update token count (RQ5)
        if "input_ids" in inputs:
            token_count = inputs["attention_mask"].sum().item()
            self.token_tracker.update(task, token_count, step=self.state.global_step)

        if return_outputs:
            return loss, outputs

        return loss

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Perform one training step.

        Args:
            model: Model
            inputs: Batch inputs

        Returns:
            Loss value
        """
        model.train()

        if self.use_pcgrad and self.pcgrad_optimizer is not None:
            # PCGrad: compute per-task losses separately
            # NOTE: This is a simplified version
            # Full implementation would batch multiple tasks per step
            task_losses = {}

            # Get task from inputs
            task = inputs.get("task")
            loss = self.compute_loss(model, inputs)
            task_losses[task] = loss

            # PCGrad step (handles backward and projection)
            self.pcgrad_optimizer.step(task_losses)

            return loss.detach()

        else:
            # Standard training step
            loss = self.compute_loss(model, inputs)

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            loss.backward()

            return loss.detach()

    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Evaluate on all tasks.

        Args:
            eval_dataset: Evaluation dataset (optional, uses self.eval_datasets)
            ignore_keys: Keys to ignore in outputs
            metric_key_prefix: Prefix for metric names

        Returns:
            Dictionary of metrics
        """
        # Evaluate on each task separately
        all_metrics = {}

        for task_name, task_dataset in self.eval_datasets.items():
            # Evaluate task
            task_metrics = super().evaluate(
                eval_dataset=task_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_{task_name}",
            )

            all_metrics.update(task_metrics)

        # Compute average metric
        task_scores = [
            metrics[f"{metric_key_prefix}_{task}_loss"]
            for task, metrics in all_metrics.items()
            if f"{metric_key_prefix}_{task}_loss" in metrics
        ]

        if task_scores:
            all_metrics[f"{metric_key_prefix}_avg_loss"] = sum(task_scores) / len(task_scores)

        return all_metrics

    def save_checkpoint(self):
        """Save checkpoint with token counts and conflict stats."""
        # Get current metrics
        metrics = self.state.log_history[-1] if self.state.log_history else {}

        # Add token counts
        metrics["tokens_per_task"] = self.token_tracker.tokens_per_task.copy()
        metrics["total_tokens"] = self.token_tracker.get_total_tokens()

        # Add conflict stats (if using PCGrad)
        if self.pcgrad_optimizer is not None:
            metrics["conflict_frequency"] = self.pcgrad_optimizer.get_conflict_frequency()

        # Save via checkpoint manager
        self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            step=self.state.global_step,
            epoch=self.state.epoch,
            metrics=metrics,
            token_counts=self.token_tracker.tokens_per_task,
        )


class TokenControlledTrainer(MultiTaskTrainer):
    """Trainer that stops when target token count is reached.

    CRITICAL for RQ5: Token-controlled baseline.

    Trains single-task model with exact token budget from multi-task run.
    This separates gains from cross-task transfer vs. data exposure.
    """

    def __init__(
        self,
        model: MultiTaskModel,
        args: TrainingArguments,
        train_dataset: Any,
        eval_dataset: Any,
        task: str,
        target_tokens: int,
        **kwargs,
    ):
        """Initialize token-controlled trainer.

        Args:
            model: Model to train
            args: Training arguments
            train_dataset: Training dataset for single task
            eval_dataset: Evaluation dataset
            task: Task name
            target_tokens: Target total tokens (from multi-task run)
            **kwargs: Additional arguments for MultiTaskTrainer
        """
        # Wrap datasets in dict for compatibility
        train_datasets = {task: train_dataset}
        eval_datasets = {task: eval_dataset}

        super().__init__(
            model=model,
            args=args,
            train_datasets=train_datasets,
            eval_datasets=eval_datasets,
            task_configs={task: {}},
            strategy="S1_token_controlled",
            **kwargs
        )

        self.task = task
        self.target_tokens = target_tokens

        print(f"\n{'='*70}")
        print(f"TOKEN-CONTROLLED TRAINING (RQ5)")
        print(f"{'='*70}")
        print(f"Task: {task}")
        print(f"Target tokens: {target_tokens:,}")
        print(f"This matches the token count from the multi-task run.")
        print(f"{'='*70}\n")

    def should_stop_training(self) -> bool:
        """Check if target token count is reached.

        Returns:
            True if should stop training
        """
        current_tokens = self.token_tracker.get_total_tokens(task=self.task)

        if current_tokens >= self.target_tokens:
            print(f"\n🎯 Target tokens reached: {current_tokens:,} / {self.target_tokens:,}")
            print(f"Stopping token-controlled training.\n")
            return True

        return False

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Training step with token count check.

        Args:
            model: Model
            inputs: Batch inputs

        Returns:
            Loss value
        """
        # Check if should stop
        if self.should_stop_training():
            self.control.should_training_stop = True

        # Normal training step
        return super().training_step(model, inputs)


def run_smoke_test_before_training(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_steps: int = 50,
) -> bool:
    """Run smoke test before starting expensive training.

    Args:
        model: Model to test
        train_loader: Training data loader
        optimizer: Optimizer
        n_steps: Number of steps to test

    Returns:
        True if smoke test passed, False otherwise
    """
    from ..utils import smoke_test

    print("\n" + "="*70)
    print("SMOKE TEST — Testing 50 steps before full training")
    print("="*70)

    result = smoke_test(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        n_steps=n_steps,
        verbose=True,
    )

    if result["passed"]:
        print("\n✅ Smoke test PASSED — proceeding to full training\n")
        return True
    else:
        print(f"\n❌ Smoke test FAILED: {result['fail_reason']}")
        print("Fix the issue before running full training.\n")
        return False
