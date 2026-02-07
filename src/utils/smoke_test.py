"""Smoke testing for training runs — catch issues in 2 minutes instead of 2 hours.

Run smoke_test() on ~100 samples for 50 steps before launching a full training run.
Checks:
- Loss decreases by >50%
- No NaN or infinite losses
- No explosion (loss >100)
- Gradients are flowing (non-zero)
- VRAM is stable
"""

import gc
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def smoke_test(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_steps: int = 50,
    max_samples: int = 100,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a smoke test on a training setup.

    Args:
        model: Model to test
        train_loader: Training data loader
        optimizer: Optimizer
        n_steps: Number of training steps to run
        max_samples: Maximum samples to use (for speed)
        device: Device to run on (default: model's device)
        verbose: Whether to print progress

    Returns:
        Dictionary with test results and pass/fail status

    Example:
        >>> smoke_result = smoke_test(model, train_loader, optimizer)
        >>> if not smoke_result['passed']:
        ...     print(f"FAIL: {smoke_result['fail_reason']}")
        ...     sys.exit(1)
        >>> print("✓ Smoke test passed! Proceeding to full training.")
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    model.to(device)

    # Results tracking
    losses = []
    gradients = []
    initial_loss = None
    final_loss = None

    # VRAM tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    else:
        initial_memory = 0.0

    if verbose:
        print(f"\n{'='*60}")
        print("SMOKE TEST")
        print(f"{'='*60}")
        print(f"Running {n_steps} steps on {max_samples} samples...")

    # Run training steps
    iterator = iter(train_loader)
    pbar = tqdm(range(n_steps), disable=not verbose, desc="Smoke test")

    samples_seen = 0
    for step in pbar:
        # Stop if we've seen enough samples
        if samples_seen >= max_samples:
            break

        try:
            batch = next(iterator)
        except StopIteration:
            # Reset iterator if we run out
            iterator = iter(train_loader)
            batch = next(iterator)

        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            batch_size = len(batch[list(batch.keys())[0]])
        else:
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            batch_size = len(batch[0])

        samples_seen += batch_size

        # Forward pass
        optimizer.zero_grad()

        try:
            outputs = model(**batch) if isinstance(batch, dict) else model(*batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        except Exception as e:
            return {
                "passed": False,
                "fail_reason": f"Forward pass failed: {str(e)}",
                "step": step,
            }

        # Check for NaN or infinite loss
        if torch.isnan(loss) or torch.isinf(loss):
            return {
                "passed": False,
                "fail_reason": f"NaN or infinite loss at step {step}",
                "loss": loss.item(),
                "step": step,
            }

        losses.append(loss.item())

        if step == 0:
            initial_loss = loss.item()

        # Backward pass
        try:
            loss.backward()
        except Exception as e:
            return {
                "passed": False,
                "fail_reason": f"Backward pass failed: {str(e)}",
                "step": step,
            }

        # Check gradient flow
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        gradients.append(grad_norm)

        # Optimizer step
        optimizer.step()

        # Update progress bar
        if verbose:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "grad_norm": f"{grad_norm:.2f}"})

    final_loss = losses[-1]

    # VRAM check
    if device.type == "cuda":
        final_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        memory_growth = final_memory - initial_memory
    else:
        final_memory = 0.0
        memory_growth = 0.0

    # Analyze results
    results = {
        "passed": True,
        "fail_reason": None,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decrease_pct": ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0,
        "avg_grad_norm": sum(gradients) / len(gradients) if gradients else 0,
        "max_loss": max(losses),
        "min_loss": min(losses),
        "initial_memory_gb": initial_memory,
        "final_memory_gb": final_memory,
        "memory_growth_gb": memory_growth,
        "steps_completed": len(losses),
    }

    # Check 1: Loss should decrease by at least 50%
    if results["loss_decrease_pct"] < 50:
        results["passed"] = False
        results["fail_reason"] = (
            f"Loss decreased by only {results['loss_decrease_pct']:.1f}% "
            f"(expected >50%). Model may not be learning."
        )

    # Check 2: Loss should be in reasonable range
    if results["max_loss"] > 100:
        results["passed"] = False
        results["fail_reason"] = (
            f"Loss exploded to {results['max_loss']:.2f}. "
            "Check learning rate and gradient clipping."
        )

    # Check 3: Gradients should be flowing
    if results["avg_grad_norm"] < 1e-6:
        results["passed"] = False
        results["fail_reason"] = (
            f"Gradients too small ({results['avg_grad_norm']:.2e}). "
            "Model may be frozen or learning rate too low."
        )

    # Check 4: VRAM should be stable (growth <1 GB for small test)
    if memory_growth > 1.0:
        results["passed"] = False
        results["fail_reason"] = (
            f"VRAM grew by {memory_growth:.2f} GB during smoke test. "
            "Possible memory leak detected."
        )

    # Cleanup
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("SMOKE TEST RESULTS")
        print(f"{'='*60}")
        print(f"Status: {'✓ PASSED' if results['passed'] else '✗ FAILED'}")
        if not results['passed']:
            print(f"Reason: {results['fail_reason']}")
        print(f"\nMetrics:")
        print(f"  Initial loss: {results['initial_loss']:.4f}")
        print(f"  Final loss:   {results['final_loss']:.4f}")
        print(f"  Decrease:     {results['loss_decrease_pct']:.1f}%")
        print(f"  Avg grad norm: {results['avg_grad_norm']:.2e}")
        if device.type == "cuda":
            print(f"\nVRAM:")
            print(f"  Initial: {results['initial_memory_gb']:.2f} GB")
            print(f"  Final:   {results['final_memory_gb']:.2f} GB")
            print(f"  Growth:  {results['memory_growth_gb']:.2f} GB")
        print(f"{'='*60}\n")

    return results


def quick_overfit_test(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_steps: int = 200,
    target_loss: float = 0.01,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> bool:
    """Test if model can overfit a single batch (sanity check).

    This tests whether the model has sufficient capacity and correct setup
    to fit a very small dataset.

    Args:
        model: Model to test
        train_loader: Training data loader
        optimizer: Optimizer
        n_steps: Number of steps to train
        target_loss: Target loss to achieve (default 0.01)
        device: Device to run on
        verbose: Whether to print progress

    Returns:
        True if model successfully overfits, False otherwise
    """
    if device is None:
        device = next(model.parameters()).device

    model.train()
    model.to(device)

    # Get a single batch
    batch = next(iter(train_loader))
    if isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}
    else:
        batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)

    if verbose:
        print(f"\nOverfit test: Training on single batch for {n_steps} steps...")

    pbar = tqdm(range(n_steps), disable=not verbose, desc="Overfit test")

    for step in pbar:
        optimizer.zero_grad()

        outputs = model(**batch) if isinstance(batch, dict) else model(*batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        loss.backward()
        optimizer.step()

        if verbose:
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        # Check if target achieved
        if loss.item() < target_loss:
            if verbose:
                print(f"✓ Achieved target loss {target_loss} at step {step}")
            return True

    final_loss = loss.item()
    if verbose:
        if final_loss < target_loss:
            print(f"✓ Overfit successful: final loss {final_loss:.6f}")
        else:
            print(f"✗ Failed to overfit: final loss {final_loss:.6f} > target {target_loss}")

    return final_loss < target_loss
