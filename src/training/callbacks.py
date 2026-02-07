"""Training callbacks for monitoring and logging.

Callbacks for budget-constrained research:
- VRAMCallback: Memory leak detection
- QuickEvalCallback: Fast evaluation on subset
- TokenLoggingCallback: Track tokens per task (RQ5)
- GradientConflictCallback: Log PCGrad conflicts (RQ4)
"""

from typing import Dict, List, Optional, Callable
from pathlib import Path

import torch
import numpy as np
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class VRAMCallback(TrainerCallback):
    """Monitor VRAM usage and detect memory leaks.

    Checks every N steps for:
    - Memory leaks (gradual growth)
    - Usage spikes
    - OOM warnings
    """

    def __init__(
        self,
        check_interval: int = 50,
        leak_threshold_gb: float = 0.5,
        alert_on_leak: bool = True,
    ):
        """Initialize VRAM callback.

        Args:
            check_interval: Check every N steps
            leak_threshold_gb: Alert if VRAM grows by this amount
            alert_on_leak: Whether to print alerts
        """
        self.check_interval = check_interval
        self.leak_threshold_gb = leak_threshold_gb
        self.alert_on_leak = alert_on_leak

        self.memory_history = []
        self.baseline_memory = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Initialize baseline memory at training start."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.baseline_memory = self._get_memory_stats()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Check VRAM after each step."""
        if state.global_step % self.check_interval == 0:
            if torch.cuda.is_available():
                stats = self._get_memory_stats()
                self.memory_history.append(stats)

                # Check for leak
                if len(self.memory_history) >= 2:
                    growth = stats["allocated_gb"] - self.memory_history[-2]["allocated_gb"]
                    if growth > self.leak_threshold_gb:
                        if self.alert_on_leak:
                            print(
                                f"⚠️  VRAM leak detected at step {state.global_step}: "
                                f"+{growth:.2f} GB"
                            )

                # Log to wandb
                if WANDB_AVAILABLE and wandb.run is not None:
                    wandb.log({
                        "vram/allocated_gb": stats["allocated_gb"],
                        "vram/reserved_gb": stats["reserved_gb"],
                        "vram/peak_gb": stats["peak_gb"],
                    }, step=state.global_step)

    def _get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
            "peak_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        }


class QuickEvalCallback(TrainerCallback):
    """Quick evaluation on subset of dev data.

    Evaluates on 500 samples every N steps for fast feedback.
    Full evaluation is expensive; this provides training signal.
    """

    def __init__(
        self,
        eval_fn: Callable,
        eval_dataset,
        eval_interval: int = 200,
        num_samples: int = 500,
    ):
        """Initialize quick eval callback.

        Args:
            eval_fn: Function that takes (model, dataset) and returns metrics
            eval_dataset: Full evaluation dataset (will be subsampled)
            eval_interval: Evaluate every N steps
            num_samples: Number of samples to evaluate on
        """
        self.eval_fn = eval_fn
        self.eval_dataset = eval_dataset
        self.eval_interval = eval_interval
        self.num_samples = num_samples

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Run quick evaluation."""
        if state.global_step % self.eval_interval == 0 and model is not None:
            # Sample subset
            indices = np.random.choice(
                len(self.eval_dataset),
                size=min(self.num_samples, len(self.eval_dataset)),
                replace=False
            )
            subset = torch.utils.data.Subset(self.eval_dataset, indices)

            # Evaluate
            model.eval()
            with torch.no_grad():
                metrics = self.eval_fn(model, subset)
            model.train()

            # Log
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    f"quick_eval/{k}": v
                    for k, v in metrics.items()
                }, step=state.global_step)

            print(f"Quick eval @ step {state.global_step}: {metrics}")


class TokenLoggingCallback(TrainerCallback):
    """Log cumulative tokens per task for RQ5.

    This is CRITICAL for the token-controlled baseline.
    Must track exactly how many tokens are seen per task.
    """

    def __init__(self, token_tracker):
        """Initialize token logging callback.

        Args:
            token_tracker: TokenTracker instance from data module
        """
        self.token_tracker = token_tracker

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log token counts."""
        if WANDB_AVAILABLE and wandb.run is not None:
            # Log per-task token counts
            for task, count in self.token_tracker.tokens_per_task.items():
                wandb.log({
                    f"tokens/{task}": count,
                }, step=state.global_step)

            # Log total tokens
            wandb.log({
                "tokens/total": self.token_tracker.get_total_tokens(),
            }, step=state.global_step)

            # Log token distribution
            distribution = self.token_tracker.get_token_distribution()
            for task, pct in distribution.items():
                wandb.log({
                    f"token_pct/{task}": pct * 100,
                }, step=state.global_step)


class GradientConflictCallback(TrainerCallback):
    """Log PCGrad conflict frequency for RQ4.

    Tracks how often tasks have conflicting gradients.
    This informs negative transfer analysis.
    """

    def __init__(self, pcgrad_optimizer, log_interval: int = 100):
        """Initialize gradient conflict callback.

        Args:
            pcgrad_optimizer: PCGradOptimizer instance
            log_interval: Log conflicts every N steps
        """
        self.pcgrad_optimizer = pcgrad_optimizer
        self.log_interval = log_interval

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log conflict statistics."""
        if state.global_step % self.log_interval == 0:
            if hasattr(self.pcgrad_optimizer, 'get_conflict_frequency'):
                conflict_freq = self.pcgrad_optimizer.get_conflict_frequency()

                # Log to wandb
                if WANDB_AVAILABLE and wandb.run is not None:
                    for task_i, conflicts in conflict_freq.items():
                        for task_j, freq in conflicts.items():
                            wandb.log({
                                f"conflicts/{task_i}_vs_{task_j}": freq,
                            }, step=state.global_step)

                    # Log total conflicts
                    total = self.pcgrad_optimizer.get_total_conflicts()
                    wandb.log({
                        "conflicts/total": total,
                    }, step=state.global_step)


class LossExplosionCallback(TrainerCallback):
    """Detect loss explosions and NaN values.

    Stops training if loss becomes NaN or explodes beyond threshold.
    """

    def __init__(self, explosion_threshold: float = 100.0):
        """Initialize loss explosion callback.

        Args:
            explosion_threshold: Stop if loss exceeds this value
        """
        self.explosion_threshold = explosion_threshold

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Check for loss explosions."""
        if logs is not None and "loss" in logs:
            loss = logs["loss"]

            # Check for NaN
            if np.isnan(loss):
                print(f"❌ Loss is NaN at step {state.global_step}. Stopping training.")
                control.should_training_stop = True

            # Check for explosion
            elif loss > self.explosion_threshold:
                print(
                    f"❌ Loss exploded to {loss:.2f} at step {state.global_step}. "
                    f"Stopping training."
                )
                control.should_training_stop = True


class SpeedCallback(TrainerCallback):
    """Track training speed and estimate time remaining.

    Logs samples/second and ETA to completion.
    """

    def __init__(self, total_steps: int):
        """Initialize speed callback.

        Args:
            total_steps: Total training steps (for ETA)
        """
        self.total_steps = total_steps
        self.step_times = []
        self.start_time = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Record start time."""
        import time
        self.start_time = time.time()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Track step time."""
        import time
        current_time = time.time()

        if len(self.step_times) > 0:
            step_duration = current_time - self.last_step_time
            self.step_times.append(step_duration)

            # Keep only last 100 steps for rolling average
            if len(self.step_times) > 100:
                self.step_times.pop(0)

        self.last_step_time = current_time

        # Log speed every 50 steps
        if state.global_step % 50 == 0 and len(self.step_times) > 0:
            avg_step_time = np.mean(self.step_times)
            steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0

            # Estimate ETA
            remaining_steps = self.total_steps - state.global_step
            remaining_seconds = remaining_steps * avg_step_time
            eta_hours = remaining_seconds / 3600
            eta_minutes = (remaining_seconds % 3600) / 60

            # Log
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({
                    "speed/steps_per_sec": steps_per_sec,
                    "speed/eta_hours": eta_hours,
                }, step=state.global_step)

            print(
                f"Speed: {steps_per_sec:.2f} steps/sec, "
                f"ETA: {int(eta_hours)}h {int(eta_minutes)}m"
            )


class CheckpointCallback(TrainerCallback):
    """Enhanced checkpointing with Kaggle safety.

    Saves:
    - LoRA adapters (not full model)
    - Optimizer state
    - Token counts (RQ5)
    - Conflict stats (RQ4)
    - Best metric
    """

    def __init__(
        self,
        checkpoint_manager,
        save_interval: int = 200,
        metric_name: str = "eval_f1",
        greater_is_better: bool = True,
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_manager: CheckpointManager instance
            save_interval: Save every N steps
            metric_name: Metric to track for best model
            greater_is_better: Whether higher metric is better
        """
        self.checkpoint_manager = checkpoint_manager
        self.save_interval = save_interval
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = float('-inf') if greater_is_better else float('inf')

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        optimizer=None,
        **kwargs,
    ):
        """Save checkpoint periodically."""
        if state.global_step % self.save_interval == 0 and model is not None:
            # Save checkpoint
            self.checkpoint_manager.save(
                model=model,
                optimizer=optimizer,
                step=state.global_step,
                epoch=state.epoch,
                metrics=state.log_history[-1] if state.log_history else {},
            )

            print(f"💾 Checkpoint saved at step {state.global_step}")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        model=None,
        **kwargs,
    ):
        """Save best model based on eval metric."""
        if metrics is not None and self.metric_name in metrics and model is not None:
            current_metric = metrics[self.metric_name]

            is_best = False
            if self.greater_is_better:
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
            else:
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    is_best = True

            if is_best:
                # Save as best model
                self.checkpoint_manager.save(
                    model=model,
                    step=state.global_step,
                    epoch=state.epoch,
                    metrics=metrics,
                )
                print(
                    f"🏆 New best model! {self.metric_name}={current_metric:.4f} "
                    f"at step {state.global_step}"
                )
