"""Quick verification script for Phase 0 installation.

Run this to verify all modules import correctly and core functionality works.
"""

import sys
from pathlib import Path

# Fix Windows Unicode issue
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Test that all modules import successfully."""
    print("Testing imports...")

    try:
        # Data module
        from data import (
            UnifiedSample,
            BaseTaskDataset,
            TaskRegistry,
            MultiTaskBatchSampler,
            TokenTracker,
            NERCollator,
            SpanCollator,
            RECollator,
            QACollator,
        )
        print("  ✓ data module")

        # Utils module
        from utils import (
            find_optimal_batch_size,
            VRAMMonitor,
            CheckpointManager,
            smoke_test,
        )
        print("  ✓ utils module")

        # Models module
        from models import load_model, get_model_config
        print("  ✓ models module")

        # Results module
        from results import (
            ExperimentResult,
            TrainingStats,
            TaskMetrics,
            EfficiencyStats,
            CalibrationStats,
            ContaminationResult,
            GradientConflictStats,
            ResultsManager,
        )
        print("  ✓ results module")

        return True

    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_unified_sample():
    """Test UnifiedSample creation."""
    print("\nTesting UnifiedSample...")

    from data import UnifiedSample

    try:
        sample = UnifiedSample(
            task="semeval2014t7",
            task_type="ner",
            task_level="token",
            input_text="Patient has abdominal pain.",
            labels=["O", "O", "B-Disorder", "I-Disorder", "O"],
            metadata={"doc_id": "123"},
            token_count=7,
        )
        print(f"  ✓ Created sample: {sample.task}")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_token_tracker():
    """Test TokenTracker functionality."""
    print("\nTesting TokenTracker...")

    from data import TokenTracker

    try:
        tracker = TokenTracker(tasks=["task1", "task2", "task3"])

        # Update with some tokens
        tracker.update("task1", 1000, step=1)
        tracker.update("task2", 500, step=1)
        tracker.update("task1", 1500, step=2)

        total = tracker.get_total_tokens()
        dist = tracker.get_token_distribution()

        print(f"  ✓ Total tokens: {total}")
        print(f"  ✓ Distribution: {dist}")

        assert total == 3000, f"Expected 3000 tokens, got {total}"
        assert tracker.tokens_per_task["task1"] == 2500

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def test_results_manager():
    """Test ResultsManager."""
    print("\nTesting ResultsManager...")

    from results import (
        ResultsManager,
        ExperimentResult,
        TrainingStats,
        TaskMetrics,
    )
    import tempfile
    import shutil

    try:
        # Create temp directory
        temp_dir = tempfile.mkdtemp()

        manager = ResultsManager(results_dir=temp_dir)

        # Create a sample result
        result = ExperimentResult(
            experiment_id="test_S1_gemma2b_task1_seed42_20260207",
            phase="phase2",
            strategy="S1",
            model_name="gemma2b",
            task="semeval2014t7",
            seed=42,
            config={"lr": 1e-4},
            git_commit="abc123",
            training=TrainingStats(
                total_steps=1000,
                total_epochs=3.0,
                tokens_per_task={"semeval2014t7": 50000},
                total_tokens=50000,
                best_step=800,
                best_epoch=2.5,
                training_time_seconds=3600,
                peak_vram_gb=8.5,
                samples_per_second=10.0,
                final_train_loss=0.25,
            ),
            metrics={
                "semeval2014t7": TaskMetrics(
                    primary_metric=0.85,
                    primary_metric_name="f1",
                    secondary_metrics={"precision": 0.83, "recall": 0.87},
                )
            },
        )

        # Save result
        path = manager.save_result(result)
        print(f"  ✓ Saved result to: {path.name}")

        # Load result
        loaded = manager.load_result(result.experiment_id)
        assert loaded is not None, "Failed to load result"
        print(f"  ✓ Loaded result: {loaded.experiment_id}")

        # Query results
        results = manager.load_all_results(phase="phase2", strategy="S1")
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        print(f"  ✓ Query returned {len(results)} result(s)")

        # Cleanup
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_config():
    """Test model config generation."""
    print("\nTesting model config...")

    from models import get_model_config

    try:
        config = get_model_config("phi3_mini")
        print(f"  ✓ Config for phi3_mini:")
        print(f"    - Size: {config.model_size_b}B params")
        print(f"    - Quantization: {config.use_quantization}")
        print(f"    - Gradient checkpointing: {config.use_gradient_checkpointing}")

        assert config.model_size_b == 3.8
        assert not config.use_quantization  # <4B, no quantization

        config_large = get_model_config("qwen25_7b")
        assert config_large.use_quantization  # >4B, auto-quantize

        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("PHASE 0 INSTALLATION VERIFICATION")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("UnifiedSample", test_unified_sample),
        ("TokenTracker", test_token_tracker),
        ("ResultsManager", test_results_manager),
        ("ModelConfig", test_model_config),
    ]

    results = []
    for name, test_func in tests:
        passed = test_func()
        results.append((name, passed))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n✅ All tests passed! Phase 0 installation is complete.")
        return 0
    else:
        print("\n❌ Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
