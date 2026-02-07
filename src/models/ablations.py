"""Architecture ablation variants (A1-A4) with enforced parameter parity.

PARAMETER PARITY IS MANDATORY: All variants must have comparable trainable
parameters (within 5%).

Ablation Matrix:
    A1: Shared only (r=24) — Tests shared representation capacity
    A2: Private only (r=24 per task) — Tests task-specific capacity
    A3: Shared (r=16) + Private (r=8), no fusion — Tests composition
    A4: Shared (r=16) + Private (r=8) + Attention fusion — Full model

Goal: Empirically justify each architectural component.
"""

from typing import Dict, List, Optional, Any

import torch.nn as nn
from peft import LoraConfig, get_peft_model

from .adapters import (
    SharedPrivateLoRA,
    verify_parameter_parity,
    count_trainable_params,
)
from .multitask_model import MultiTaskModel


def calculate_ablation_ranks(
    base_model: nn.Module,
    num_tasks: int,
    target_params: Optional[int] = None,
    num_target_modules: int = 7,
) -> Dict[str, Dict[str, int]]:
    """Calculate LoRA ranks for each ablation to achieve parameter parity.

    This function ensures all ablations (A1-A4) have approximately the same
    number of trainable parameters.

    For a typical 3B model with 32 layers:
        - Each LoRA layer adds: 2 * hidden_size * rank per target module
        - With 7 target modules: 7 * 2 * hidden_size * rank * num_layers

    Args:
        base_model: Base pretrained model
        num_tasks: Number of tasks (affects A2 private-only)
        target_params: Target trainable parameter count (default: auto-calculate)
        num_target_modules: Number of modules with LoRA applied

    Returns:
        Dictionary with rank configurations for each ablation
    """
    hidden_size = base_model.config.hidden_size
    num_layers = base_model.config.num_hidden_layers

    # Estimate params per rank per layer
    params_per_rank_per_layer = 2 * hidden_size * num_target_modules

    # Total params per rank
    params_per_rank = params_per_rank_per_layer * num_layers

    # If no target specified, aim for ~25M trainable params
    if target_params is None:
        target_params = 25_000_000

    # Calculate ranks to match target
    # A1: Shared only
    a1_shared_rank = int(target_params / params_per_rank)

    # A2: Private only (per task)
    # Each task has separate adapter, so divide target by num_tasks
    a2_private_rank = int(target_params / (params_per_rank * num_tasks))

    # A3 & A4: Shared + Private
    # Split between shared and private (2:1 ratio to match paper)
    # shared contributes 2/3, private contributes 1/3 per task
    # Total: shared + num_tasks * private = target
    # With ratio: shared = 2 * private
    # So: 2 * private + num_tasks * private = target
    # private * (2 + num_tasks) = target
    a34_combined_rank_budget = target_params / params_per_rank
    a34_private_rank = int(a34_combined_rank_budget / (2 + num_tasks))
    a34_shared_rank = 2 * a34_private_rank

    return {
        "A1": {
            "shared_rank": a1_shared_rank,
            "private_rank": 0,
        },
        "A2": {
            "shared_rank": 0,
            "private_rank": a2_private_rank,
        },
        "A3": {
            "shared_rank": a34_shared_rank,
            "private_rank": a34_private_rank,
        },
        "A4": {
            "shared_rank": a34_shared_rank,
            "private_rank": a34_private_rank,
        },
    }


def create_ablation_variant(
    variant: str,
    base_model: nn.Module,
    task_configs: Dict[str, Dict[str, Any]],
    rank_config: Optional[Dict[str, int]] = None,
) -> MultiTaskModel:
    """Create an ablation variant (A1-A4).

    Args:
        variant: Ablation variant ("A1", "A2", "A3", "A4")
        base_model: Base pretrained model
        task_configs: Task configurations (same as MultiTaskModel)
        rank_config: Optional rank configuration (auto-calculated if None)

    Returns:
        MultiTaskModel configured for the ablation
    """
    if variant not in ["A1", "A2", "A3", "A4"]:
        raise ValueError(f"Unknown variant: {variant}. Valid: A1, A2, A3, A4")

    task_names = list(task_configs.keys())
    num_tasks = len(task_names)

    # Calculate ranks if not provided
    if rank_config is None:
        all_ranks = calculate_ablation_ranks(base_model, num_tasks)
        rank_config = all_ranks[variant]

    # Create adapter config based on variant
    if variant == "A1":
        # Shared only (r=24)
        adapter_config = {
            "rank": rank_config["shared_rank"],
            "alpha": 32,
            "dropout": 0.1,
        }
        strategy = "S2"  # Use shared LoRA strategy

    elif variant == "A2":
        # Private only (r=24 per task)
        # Use S1 strategy (single-task) but with matched ranks
        adapter_config = {
            "rank": rank_config["private_rank"],
            "alpha": 32,
            "dropout": 0.1,
        }
        strategy = "S1"  # Single-task LoRA

    elif variant == "A3":
        # Shared + Private, no fusion
        adapter_config = {
            "shared_rank": rank_config["shared_rank"],
            "private_rank": rank_config["private_rank"],
            "alpha": 32,
            "dropout": 0.1,
            "fusion_type": "none",  # No fusion
        }
        strategy = "S3a"

    elif variant == "A4":
        # Shared + Private + Attention fusion (full model)
        adapter_config = {
            "shared_rank": rank_config["shared_rank"],
            "private_rank": rank_config["private_rank"],
            "alpha": 32,
            "dropout": 0.1,
            "fusion_type": "attention",  # Full fusion
        }
        strategy = "S3a"

    # Create model
    model = MultiTaskModel(
        base_model=base_model,
        strategy=strategy,
        task_configs=task_configs,
        adapter_config=adapter_config,
    )

    return model


def create_all_ablations(
    base_model: nn.Module,
    task_configs: Dict[str, Dict[str, Any]],
    verify_parity: bool = True,
) -> Dict[str, MultiTaskModel]:
    """Create all four ablation variants (A1-A4).

    Args:
        base_model: Base pretrained model
        task_configs: Task configurations
        verify_parity: Whether to verify parameter parity (default: True)

    Returns:
        Dictionary mapping variant names to models

    Raises:
        AssertionError if parameter parity check fails
    """
    num_tasks = len(task_configs)

    # Calculate matched ranks
    rank_configs = calculate_ablation_ranks(base_model, num_tasks)

    # Create all variants
    models = {}
    for variant in ["A1", "A2", "A3", "A4"]:
        models[variant] = create_ablation_variant(
            variant=variant,
            base_model=base_model,
            task_configs=task_configs,
            rank_config=rank_configs[variant],
        )

    # Verify parameter parity
    if verify_parity:
        verify_parameter_parity(models, tolerance=0.05)

    return models


def print_ablation_summary(models: Dict[str, MultiTaskModel]):
    """Print summary of ablation variants.

    Args:
        models: Dictionary of ablation models
    """
    print(f"\n{'='*70}")
    print("ABLATION VARIANTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<10} {'Description':<35} {'Trainable Params':>15}")
    print(f"{'-'*70}")

    descriptions = {
        "A1": "Shared LoRA only",
        "A2": "Private LoRA per task only",
        "A3": "Shared + Private (no fusion)",
        "A4": "Shared + Private + Attention fusion",
    }

    for variant in ["A1", "A2", "A3", "A4"]:
        if variant in models:
            model = models[variant]
            params = count_trainable_params(model)
            desc = descriptions[variant]
            print(f"{variant:<10} {desc:<35} {params:>15,}")

    print(f"{'='*70}\n")


# Unit test for ablation variants
def test_ablation_parity():
    """Unit test to verify ablation parameter parity.

    This test creates all four ablations and verifies they have
    matched parameter counts.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    # Create a small test model (for quick testing)
    config = AutoConfig.from_pretrained("gpt2")
    config.num_hidden_layers = 12  # Smaller for testing
    base_model = AutoModelForCausalLM.from_config(config)

    # Test task configs
    task_configs = {
        "task1": {"task_type": "ner", "num_labels": 9},
        "task2": {"task_type": "ner", "num_labels": 9},
        "task3": {"task_type": "re", "num_labels": 5},
    }

    # Create all ablations
    try:
        models = create_all_ablations(
            base_model=base_model,
            task_configs=task_configs,
            verify_parity=True,
        )

        print_ablation_summary(models)

        print("✅ Ablation parity test PASSED")
        return True

    except AssertionError as e:
        print(f"❌ Ablation parity test FAILED: {e}")
        return False


if __name__ == "__main__":
    # Run test
    test_ablation_parity()
