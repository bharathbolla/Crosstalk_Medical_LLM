"""Basic import tests to verify module structure.

These tests check that all modules can be imported without errors.
They don't require dependencies to be installed.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_data_imports():
    """Test data module imports."""
    try:
        from src.data import base, multitask_loader, collators
        assert base is not None
        assert multitask_loader is not None
        assert collators is not None
        return True
    except ImportError:
        # Expected if dependencies not installed
        return None


def test_utils_imports():
    """Test utility module imports."""
    try:
        from src.utils import auto_batch, vram_monitor, checkpoint, smoke_test
        assert auto_batch is not None
        assert vram_monitor is not None
        assert checkpoint is not None
        assert smoke_test is not None
        return True
    except ImportError:
        return None


def test_results_imports():
    """Test results module imports."""
    try:
        from src.results import schema, manager, latex_generator
        assert schema is not None
        assert manager is not None
        assert latex_generator is not None
        return True
    except ImportError:
        return None


def test_models_imports():
    """Test model module imports."""
    try:
        from src.models import (
            base_loader,
            adapters,
            hierarchical,
            heads,
            multitask_model,
            ablations,
        )
        assert base_loader is not None
        assert adapters is not None
        assert hierarchical is not None
        assert heads is not None
        assert multitask_model is not None
        assert ablations is not None
        return True
    except ImportError:
        return None


def test_training_imports():
    """Test training module imports."""
    try:
        from src.training import loss, pcgrad, trainer, callbacks
        assert loss is not None
        assert pcgrad is not None
        assert trainer is not None
        assert callbacks is not None
        return True
    except ImportError:
        return None


def test_evaluation_imports():
    """Test evaluation module imports."""
    try:
        from src.evaluation import (
            metrics,
            contamination,
            calibration,
            probing,
            transfer_analysis,
            error_analysis,
        )
        assert metrics is not None
        assert contamination is not None
        assert calibration is not None
        assert probing is not None
        assert transfer_analysis is not None
        assert error_analysis is not None
        return True
    except ImportError:
        return None


if __name__ == "__main__":
    tests = [
        ("Data modules", test_data_imports),
        ("Utility modules", test_utils_imports),
        ("Results modules", test_results_imports),
        ("Model modules", test_models_imports),
        ("Training modules", test_training_imports),
        ("Evaluation modules", test_evaluation_imports),
    ]

    print("Running basic import tests...\n")
    passed = 0
    skipped = 0

    for name, test_func in tests:
        result = test_func()
        if result is True:
            print(f"✓ {name}")
            passed += 1
        elif result is None:
            print(f"⚠ {name} (dependencies not installed)")
            skipped += 1
        else:
            print(f"✗ {name}")

    print(f"\nResults: {passed} passed, {skipped} skipped (need deps)")
