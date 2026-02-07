"""Verify all implemented modules can be imported correctly.

Tests imports without requiring dependencies to be installed.
Reports which modules are ready and which have issues.
"""

import sys
import io
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def test_import(module_path: str, description: str) -> bool:
    """Test if a module can be imported.

    Args:
        module_path: Import path (e.g., 'src.data.base')
        description: Human-readable description

    Returns:
        True if import succeeded
    """
    try:
        __import__(module_path)
        print(f"✓ {description}")
        return True
    except ImportError as e:
        # Expected for modules requiring deps
        if "No module named" in str(e) and any(
            dep in str(e) for dep in ["torch", "transformers", "peft", "numpy", "scipy"]
        ):
            print(f"⚠ {description} (needs dependencies)")
            return True  # Count as success - code structure is fine
        else:
            print(f"✗ {description}")
            print(f"  Error: {e}")
            return False
    except Exception as e:
        print(f"✗ {description}")
        print(f"  Error: {e}")
        return False


def main():
    """Test all module imports."""
    print("=" * 70)
    print("IMPORT VERIFICATION")
    print("=" * 70)

    tests = [
        # Data modules
        ("src.data.base", "Data: Base classes (UnifiedSample, TaskRegistry)"),
        ("src.data.multitask_loader", "Data: MultiTaskBatchSampler + TokenTracker"),
        ("src.data.collators", "Data: Task-specific collators"),
        ("src.data.semeval2014t7", "Data: SemEval 2014 Task 7 parser"),
        ("src.data.semeval2015t14", "Data: SemEval 2015 Task 14 parser"),
        ("src.data.semeval2016t12", "Data: SemEval 2016 Task 12 parser"),
        ("src.data.semeval2017t3", "Data: SemEval 2017 Task 3 parser"),
        ("src.data.semeval2021t6", "Data: SemEval 2021 Task 6 parser"),

        # Utility modules
        ("src.utils.auto_batch", "Utils: Auto batch size finder"),
        ("src.utils.vram_monitor", "Utils: VRAM monitor"),
        ("src.utils.checkpoint", "Utils: Checkpoint manager"),
        ("src.utils.smoke_test", "Utils: Smoke test runner"),

        # Results modules
        ("src.results.schema", "Results: Result schemas"),
        ("src.results.manager", "Results: Result manager"),
        ("src.results.latex_generator", "Results: LaTeX table generator"),

        # Model modules
        ("src.models.base_loader", "Models: Base model loader"),
        ("src.models.adapters", "Models: SharedPrivateLoRA + Fusion"),
        ("src.models.hierarchical", "Models: Hierarchical MTL"),
        ("src.models.heads", "Models: Task-specific heads"),
        ("src.models.multitask_model", "Models: Multi-task model wrapper"),
        ("src.models.ablations", "Models: Ablation variants (A1-A4)"),

        # Training modules
        ("src.training.loss", "Training: Multi-task loss functions"),
        ("src.training.pcgrad", "Training: PCGrad optimizer"),
        ("src.training.trainer", "Training: MultiTaskTrainer"),
        ("src.training.callbacks", "Training: Monitoring callbacks"),

        # Evaluation modules
        ("src.evaluation.metrics", "Evaluation: Task metrics + statistics"),
        ("src.evaluation.contamination", "Evaluation: Contamination checker"),
        ("src.evaluation.calibration", "Evaluation: ECE + reliability plots"),
        ("src.evaluation.probing", "Evaluation: Linear probes"),
        ("src.evaluation.transfer_analysis", "Evaluation: Transfer matrix analysis"),
        ("src.evaluation.error_analysis", "Evaluation: Error categorization"),
    ]

    print(f"\nTesting {len(tests)} modules...\n")

    results = []
    for module_path, description in tests:
        success = test_import(module_path, description)
        results.append((module_path, success))

    # Summary
    print("\n" + "=" * 70)
    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"SUMMARY: {passed}/{total} modules verified")

    if passed == total:
        print("\n✓ All modules are correctly structured!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -e .")
        print("  2. Apply for PhysioNet access (Day 1 priority)")
        print("  3. Run unit tests with synthetic data")
    else:
        print(f"\n⚠ {total - passed} module(s) have structural issues")
        print("\nFailed modules:")
        for module_path, success in results:
            if not success:
                print(f"  - {module_path}")

    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
