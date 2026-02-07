"""Main experiment runner using Hydra for config management.

Usage:
    # Single experiment
    python run_experiment.py model=llama3b task=semeval2017t3 strategy=s1_single

    # With experiment config
    python run_experiment.py +experiment=s3b_llama8b

    # Multi-task all tasks
    python run_experiment.py model=llama3b strategy=s3b_hierarchical task=all

    # Token-controlled baseline
    python run_experiment.py +experiment=token_controlled model=llama3b
"""

import sys
import io
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.base import TaskRegistry
from src.data.multitask_loader import MultiTaskBatchSampler, TokenTracker
from src.models.base_loader import load_model
from src.models.multitask_model import MultiTaskModel
from src.training.trainer import MultiTaskTrainer, TokenControlledTrainer
from src.training.loss import UncertaintyWeightedLoss, EqualWeightedLoss
from src.training.pcgrad import PCGradOptimizer
from src.training.callbacks import (
    VRAMCallback,
    TokenLoggingCallback,
    GradientConflictCallback,
    QuickEvalCallback,
)
from src.utils.auto_batch import find_optimal_batch_size
from src.utils.vram_monitor import VRAMMonitor
from src.utils.checkpoint import CheckpointManager
from src.utils.smoke_test import run_smoke_test
from src.results.manager import ResultManager
from src.evaluation.metrics import compute_task_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Run experiment with Hydra config."""

    # Print config
    print("=" * 80)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set seed
    torch.manual_seed(cfg.project.seed)

    # Initialize wandb
    if cfg.logging.wandb.enabled:
        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.logging.wandb.tags,
        )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # VRAM monitor
    vram_monitor = VRAMMonitor()
    vram_monitor.log_current_usage()

    # Load model
    print(f"\nLoading model: {cfg.model.name}")
    model = load_model(
        model_name=cfg.model.name,
        quantization_config=cfg.model.quantization if cfg.model.quantization.enabled else None,
        lora_config=cfg.model.lora,
        device=device,
    )

    # Load tasks
    task_registry = TaskRegistry()
    if cfg.task.name == "all":
        tasks = ["semeval2014t7", "semeval2015t14", "semeval2016t12",
                 "semeval2017t3", "semeval2021t6"]
        print(f"\nLoading all {len(tasks)} tasks")
    else:
        tasks = [cfg.task.name]
        print(f"\nLoading task: {cfg.task.name}")

    # Load datasets
    train_datasets = {}
    dev_datasets = {}
    for task in tasks:
        print(f"  Loading {task}...")
        train_datasets[task] = task_registry.get_dataset(task, split="train")
        dev_datasets[task] = task_registry.get_dataset(task, split="dev")

    # Token tracker (RQ5 critical)
    token_tracker = TokenTracker()

    # Multi-task batch sampler
    if cfg.strategy.type == "multi_task":
        batch_sampler = MultiTaskBatchSampler(
            datasets=train_datasets,
            batch_size=cfg.model.training.max_batch_size_t4,
            sampling_strategy=cfg.strategy.multitask.sampling,
            temperature=cfg.strategy.multitask.get("temperature", 2.0),
        )

    # Create multi-task model
    print(f"\nCreating model with strategy: {cfg.strategy.name}")
    multitask_model = MultiTaskModel(
        base_model=model,
        tasks=tasks,
        strategy=cfg.strategy.name.split("_")[0],  # S1, S2, S3a, S3b, etc.
        adapter_config=cfg.strategy.adapter,
        device=device,
    )

    # Loss function
    if cfg.strategy.type == "multi_task":
        if cfg.strategy.multitask.loss_weighting == "uncertainty":
            loss_fn = UncertaintyWeightedLoss(tasks)
        else:
            loss_fn = EqualWeightedLoss(tasks)
    else:
        loss_fn = None  # single-task uses standard loss

    # Optimizer
    optimizer = torch.optim.AdamW(
        multitask_model.parameters(),
        lr=cfg.strategy.training.learning_rate,
    )

    # PCGrad wrapper (if enabled)
    if cfg.strategy.get("gradient", {}).get("conflict_resolution") == "pcgrad":
        print("Using PCGrad for gradient conflict resolution")
        optimizer = PCGradOptimizer(
            optimizer=optimizer,
            model=multitask_model,
            task_names=tasks,
        )

    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(cfg.paths.checkpoints_dir),
        max_checkpoints=2,
    )

    # Callbacks
    callbacks = [
        VRAMCallback(),
        TokenLoggingCallback(token_tracker=token_tracker),
    ]

    if cfg.strategy.type == "multi_task" and cfg.strategy.get("gradient", {}).get("log_conflicts"):
        callbacks.append(GradientConflictCallback(optimizer=optimizer))

    if cfg.strategy.training.eval_every_n_steps:
        callbacks.append(
            QuickEvalCallback(
                eval_every_n_steps=cfg.strategy.training.eval_every_n_steps,
                num_samples=500,
            )
        )

    # Trainer
    token_controlled = cfg.get("token_control", {}).get("enabled", False)

    if token_controlled:
        print("Using TokenControlledTrainer (RQ5)")
        target_tokens = cfg.token_control.get("target_tokens", None)
        trainer = TokenControlledTrainer(
            model=multitask_model,
            token_tracker=token_tracker,
            target_tokens=target_tokens,
            checkpoint_manager=checkpoint_manager,
            callbacks=callbacks,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
    else:
        trainer = MultiTaskTrainer(
            model=multitask_model,
            token_tracker=token_tracker,
            checkpoint_manager=checkpoint_manager,
            callbacks=callbacks,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    trainer.train(
        train_datasets=train_datasets,
        dev_datasets=dev_datasets,
        epochs=cfg.strategy.training.epochs,
    )

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    test_datasets = {
        task: task_registry.get_dataset(task, split="test")
        for task in tasks
    }

    results = {}
    for task in tasks:
        print(f"\nEvaluating {task}...")
        predictions = trainer.predict(test_datasets[task])
        labels = [sample.labels for sample in test_datasets[task]]

        metrics = compute_task_metrics(
            task_name=task,
            predictions=predictions,
            labels=labels,
        )

        results[task] = metrics
        print(f"  {cfg.task.evaluation.primary_metric}: {metrics[cfg.task.evaluation.primary_metric]:.4f}")

    # Save results
    result_manager = ResultManager(results_dir=Path(cfg.paths.results_dir))
    result_manager.save_result(
        experiment_id=f"{cfg.model.short_name}_{cfg.strategy.name}_{cfg.task.name}",
        model_name=cfg.model.name,
        strategy=cfg.strategy.name,
        task_results=results,
        token_counts=token_tracker.tokens_per_task,
    )

    # Close wandb
    if cfg.logging.wandb.enabled:
        wandb.finish()

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
