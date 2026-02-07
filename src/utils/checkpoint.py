"""Checkpoint management for Kaggle-safe training with session death protection.

Critical principles:
1. Save LoRA adapters only (10-50 MB), not full models (2-8 GB)
2. Save every 200 steps (Kaggle sessions die without warning)
3. Keep only last 2 checkpoints (disk space limit: 20 GB)
4. Save optimizer state, scheduler state, token counts, metrics
5. Enable seamless resume from any checkpoint
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CheckpointManager:
    """Manages checkpoints with automatic cleanup and recovery.

    Design for Kaggle:
    - Default path: /kaggle/working/checkpoints/
    - Saves adapters only (PEFT models)
    - Keeps max_keep checkpoints (default 2)
    - Saves metadata: step, epoch, best_metric, token_counts
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        max_keep: int = 2,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_keep: Maximum number of checkpoints to keep (oldest deleted first)
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_keep = max_keep
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track checkpoint history
        self.checkpoint_history = []

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        token_counts: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: Model to save (PEFT model or regular PyTorch model)
            optimizer: Optimizer state (optional)
            scheduler: LR scheduler state (optional)
            step: Current training step
            epoch: Current epoch
            metrics: Dictionary of evaluation metrics
            token_counts: Dictionary of tokens per task (CRITICAL for RQ5)
            metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint directory
        """
        checkpoint_name = f"checkpoint-step-{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        # Check if it's a PEFT model (has save_pretrained method)
        if hasattr(model, 'save_pretrained'):
            # Save LoRA adapters only (10-50 MB)
            model.save_pretrained(checkpoint_path / "adapter")
        else:
            # Fallback: save full model state dict (use for small models only)
            torch.save(model.state_dict(), checkpoint_path / "model.pt")

        # Save optimizer state
        if self.save_optimizer and optimizer is not None:
            torch.save(
                optimizer.state_dict(),
                checkpoint_path / "optimizer.pt"
            )

        # Save scheduler state
        if self.save_scheduler and scheduler is not None:
            torch.save(
                scheduler.state_dict(),
                checkpoint_path / "scheduler.pt"
            )

        # Save metadata
        checkpoint_metadata = {
            "step": step,
            "epoch": epoch,
            "metrics": metrics or {},
            "token_counts": token_counts or {},
            "metadata": metadata or {},
        }

        with open(checkpoint_path / "metadata.json", "w") as f:
            json.dump(checkpoint_metadata, f, indent=2)

        # Update history
        self.checkpoint_history.append({
            "path": checkpoint_path,
            "step": step,
            "epoch": epoch,
        })

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)

        Returns:
            Metadata dictionary if checkpoint found, None otherwise
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-step-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )

        if not checkpoints:
            return None

        latest_checkpoint = checkpoints[-1]
        return self.load(latest_checkpoint, model, optimizer, scheduler)

    def load(
        self,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
    ) -> Dict[str, Any]:
        """Load a specific checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)

        Returns:
            Metadata dictionary
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load model
        adapter_path = checkpoint_path / "adapter"
        model_path = checkpoint_path / "model.pt"

        if adapter_path.exists():
            # Load PEFT adapters
            if hasattr(model, 'load_adapter'):
                model.load_adapter(adapter_path)
            elif hasattr(model, 'from_pretrained'):
                # Load adapter weights manually
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, adapter_path)
        elif model_path.exists():
            # Load full model state
            model.load_state_dict(torch.load(model_path))
        else:
            raise FileNotFoundError(
                f"No model found in checkpoint: {checkpoint_path}"
            )

        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer is not None and optimizer_path.exists():
            optimizer.load_state_dict(torch.load(optimizer_path))

        # Load scheduler state
        scheduler_path = checkpoint_path / "scheduler.pt"
        if scheduler is not None and scheduler_path.exists():
            scheduler.load_state_dict(torch.load(scheduler_path))

        # Load metadata
        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return metadata

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if len(self.checkpoint_history) <= self.max_keep:
            return

        # Sort by step
        self.checkpoint_history.sort(key=lambda x: x["step"])

        # Remove oldest checkpoints
        to_remove = self.checkpoint_history[:-self.max_keep]

        for checkpoint_info in to_remove:
            checkpoint_path = checkpoint_info["path"]
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)

        # Update history
        self.checkpoint_history = self.checkpoint_history[-self.max_keep:]

    def list_checkpoints(self) -> list:
        """List all available checkpoints.

        Returns:
            List of checkpoint paths sorted by step number
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint-step-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        return checkpoints

    def get_best_checkpoint(
        self,
        metric_name: str,
        higher_is_better: bool = True,
    ) -> Optional[Path]:
        """Find checkpoint with best metric value.

        Args:
            metric_name: Metric to compare (e.g., "eval_f1")
            higher_is_better: Whether higher values are better

        Returns:
            Path to best checkpoint, or None if no checkpoints
        """
        checkpoints = self.list_checkpoints()

        if not checkpoints:
            return None

        best_checkpoint = None
        best_value = float('-inf') if higher_is_better else float('inf')

        for checkpoint_path in checkpoints:
            metadata_path = checkpoint_path / "metadata.json"
            if not metadata_path.exists():
                continue

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            metric_value = metadata.get("metrics", {}).get(metric_name)
            if metric_value is None:
                continue

            if higher_is_better:
                if metric_value > best_value:
                    best_value = metric_value
                    best_checkpoint = checkpoint_path
            else:
                if metric_value < best_value:
                    best_value = metric_value
                    best_checkpoint = checkpoint_path

        return best_checkpoint

    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Get metadata for a specific checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Metadata dictionary
        """
        metadata_path = Path(checkpoint_path) / "metadata.json"

        if not metadata_path.exists():
            return {}

        with open(metadata_path, "r") as f:
            return json.load(f)

    def __repr__(self) -> str:
        num_checkpoints = len(self.list_checkpoints())
        return (
            f"CheckpointManager(\n"
            f"  dir={self.checkpoint_dir},\n"
            f"  max_keep={self.max_keep},\n"
            f"  num_checkpoints={num_checkpoints}\n"
            f")"
        )
