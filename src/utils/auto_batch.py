"""Automatic batch size finder for GPU memory optimization.

Critical for Kaggle T4 (16 GB VRAM) — avoid OOM failures by finding the
maximum batch size that fits in memory with target utilization.
"""

import gc
from typing import Callable, Optional

import torch
import torch.nn as nn


def find_optimal_batch_size(
    model: nn.Module,
    sample_batch_fn: Callable[[int], dict],
    min_bs: int = 1,
    max_bs: int = 64,
    target_util: float = 0.85,
    verbose: bool = True,
) -> int:
    """Find maximum batch size that fits in GPU memory using binary search.

    Args:
        model: PyTorch model to test
        sample_batch_fn: Function that takes batch_size and returns a sample batch dict
                        with keys like 'input_ids', 'attention_mask', etc.
        min_bs: Minimum batch size to try
        max_bs: Maximum batch size to try
        target_util: Target GPU utilization (0.85 = 85% of VRAM)
        verbose: Whether to print progress

    Returns:
        Optimal batch size (largest that fits within target_util)

    Example:
        >>> def make_batch(bs):
        ...     return {
        ...         'input_ids': torch.randint(0, 1000, (bs, 512)).cuda(),
        ...         'attention_mask': torch.ones(bs, 512).cuda(),
        ...         'labels': torch.randint(0, 2, (bs,)).cuda(),
        ...     }
        >>> optimal_bs = find_optimal_batch_size(model, make_batch)
        >>> print(f"Use batch size: {optimal_bs}")
    """
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available. Defaulting to batch_size=8")
        return 8

    device = next(model.parameters()).device
    if device.type != "cuda":
        if verbose:
            print(f"Model on {device}, not CUDA. Defaulting to batch_size=8")
        return 8

    # Get GPU info
    gpu_props = torch.cuda.get_device_properties(device)
    total_memory = gpu_props.total_memory / (1024 ** 3)  # Convert to GB

    if verbose:
        print(f"GPU: {gpu_props.name}")
        print(f"Total VRAM: {total_memory:.2f} GB")
        print(f"Target utilization: {target_util * 100:.0f}%")
        print(f"Searching batch size in range [{min_bs}, {max_bs}]...")

    model.train()  # Ensure we're testing training memory usage
    last_successful_bs = min_bs
    low, high = min_bs, max_bs

    # Binary search for maximum batch size
    while low <= high:
        mid = (low + high) // 2

        # Clear cache before testing
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        try:
            # Create sample batch
            batch = sample_batch_fn(mid)

            # Move batch to device if not already there
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            # Backward pass (simulates training)
            loss.backward()

            # Check memory usage
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            utilization = peak_memory / total_memory

            if verbose:
                print(f"  bs={mid}: {peak_memory:.2f} GB ({utilization * 100:.1f}%) - OK")

            # Success! Try larger batch size
            if utilization <= target_util:
                last_successful_bs = mid
                low = mid + 1
            else:
                # Too much memory, try smaller
                high = mid - 1

            # Cleanup
            model.zero_grad()
            del batch, outputs, loss

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if verbose:
                    print(f"  bs={mid}: OOM")
                # OOM, try smaller batch size
                high = mid - 1
            else:
                # Different error, re-raise
                raise e

        finally:
            # Aggressive cleanup
            gc.collect()
            torch.cuda.empty_cache()

    # Final cleanup
    model.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()

    if verbose:
        print(f"\n✓ Optimal batch size: {last_successful_bs}")
        print(f"  (Run auto-batch before every new model to avoid OOM)")

    return last_successful_bs


def estimate_max_sequence_length(
    model: nn.Module,
    batch_size: int,
    sample_fn: Callable[[int, int], dict],
    max_length: int = 2048,
    target_util: float = 0.85,
    verbose: bool = True,
) -> int:
    """Estimate maximum sequence length for a given batch size.

    Useful for determining if you can afford longer sequences with smaller batches.

    Args:
        model: PyTorch model
        batch_size: Fixed batch size to test
        sample_fn: Function that takes (batch_size, seq_length) and returns batch
        max_length: Maximum sequence length to test
        target_util: Target GPU utilization
        verbose: Whether to print progress

    Returns:
        Maximum sequence length that fits within target_util
    """
    if not torch.cuda.is_available():
        return 512  # Default fallback

    device = next(model.parameters()).device
    model.train()

    # Binary search over sequence lengths
    low, high = 128, max_length
    last_successful_len = 128

    while low <= high:
        mid = (low + high) // 2

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        try:
            batch = sample_fn(batch_size, mid)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            loss.backward()

            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            utilization = peak_memory / total_memory

            if verbose:
                print(f"  seq_len={mid}: {peak_memory:.2f} GB ({utilization * 100:.1f}%)")

            if utilization <= target_util:
                last_successful_len = mid
                low = mid + 128
            else:
                high = mid - 128

            model.zero_grad()
            del batch, outputs, loss

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = mid - 128
            else:
                raise e

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    if verbose:
        print(f"\n✓ Max sequence length: {last_successful_len} (batch_size={batch_size})")

    return last_successful_len
