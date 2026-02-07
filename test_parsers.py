"""Test script to verify all 8 parsers work correctly."""

import sys
import io
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data import (
    BC2GMDataset,
    JNLPBADataset,
    ChemProtDataset,
    DDIDataset,
    GADDataset,
    HoCDataset,
    PubMedQADataset,
    BIOSSESDataset,
    TaskRegistry
)


def test_parser(parser_class, task_name, data_path="data/raw"):
    """Test a single parser."""
    print(f"\n{'='*60}")
    print(f"Testing {task_name.upper()}")
    print(f"{'='*60}")

    try:
        # Load dataset
        dataset = parser_class(
            data_path=Path(data_path),
            split="train"
        )

        # Check basic properties
        print(f"✓ Loaded {len(dataset)} samples")
        print(f"✓ Task: {dataset.task_name}")
        print(f"✓ Type: {dataset.task_type}")
        print(f"✓ Level: {dataset.task_level}")

        # Check first sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nFirst sample:")
            print(f"  Text: {sample.input_text[:100]}...")
            print(f"  Labels: {str(sample.labels)[:100]}...")
            print(f"  Metadata keys: {list(sample.metadata.keys())}")

        # Check label schema
        try:
            schema = dataset.get_label_schema()
            print(f"✓ Label schema: {len(schema)} labels")
        except NotImplementedError:
            print(f"  (No label schema - task-specific)")

        print(f"\n[SUCCESS] {task_name} parser works!")
        return True

    except Exception as e:
        print(f"\n[FAILED] {task_name} parser failed!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all parsers."""
    print("=" * 60)
    print("PARSER TESTING SUITE")
    print("=" * 60)
    print("Testing 8 medical NLP dataset parsers...")

    results = {}

    # Test each parser
    parsers = [
        (BC2GMDataset, "bc2gm"),
        (JNLPBADataset, "jnlpba"),
        (ChemProtDataset, "chemprot"),
        (DDIDataset, "ddi"),
        (GADDataset, "gad"),
        (HoCDataset, "hoc"),
        (PubMedQADataset, "pubmedqa"),
        (BIOSSESDataset, "biosses"),
    ]

    for parser_class, task_name in parsers:
        results[task_name] = test_parser(parser_class, task_name)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    success_count = sum(results.values())
    total_count = len(results)

    print(f"\nParsers tested: {success_count}/{total_count}")

    for task_name, success in results.items():
        status = "[OK]" if success else "[X]"
        print(f"  {status} {task_name}")

    # Check TaskRegistry
    print(f"\nRegistered tasks: {TaskRegistry.list_tasks()}")

    if success_count == total_count:
        print(f"\n[SUCCESS] All {total_count} parsers work!")
        print("\nYou can now:")
        print("  1. Update configs in configs/task/")
        print("  2. Run experiments: python scripts/run_experiment.py")
        return 0
    else:
        print(f"\n[FAILED] {total_count - success_count} parser(s) failed")
        print("  Fix errors above before running experiments")
        return 1


if __name__ == "__main__":
    sys.exit(main())
