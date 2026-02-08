"""
Master Test Runner
Runs all unit tests and generates comprehensive report.

Usage: python run_all_tests.py
"""

import sys
import io
import subprocess
import time
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def run_test(test_file, description):
    """Run a single test file."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"File: {test_file}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(result.stdout)
            print(f"\n✅ PASSED in {elapsed:.2f}s")
            return True
        else:
            print(result.stdout)
            print(result.stderr)
            print(f"\n❌ FAILED in {elapsed:.2f}s")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Run all tests."""

    print("="*60)
    print("COMPREHENSIVE TEST SUITE")
    print("Medical Multi-Task Learning Project")
    print("="*60)
    print("\nThis runs ALL unit tests to verify:")
    print("  - Data loading for all 8 datasets")
    print("  - Tokenization (BERT, RoBERTa, BioBERT)")
    print("  - Label alignment for NER")
    print("  - Model loading for all task types")
    print("  - UniversalMedicalDataset for all tasks")
    print("  - Pickle file integrity")
    print("\nEstimated time: 5-10 minutes")
    print("="*60)

    # Define all tests
    tests = [
        ('test_pickle_load.py', 'Pickle File Loading'),
        ('verify_all_datasets.py', 'Dataset Verification'),
        ('tests/test_data_loading.py', 'Data Loading Tests'),
        ('tests/test_tokenization.py', 'Tokenization Tests'),
        ('tests/test_label_alignment.py', 'Label Alignment Tests'),
        ('tests/test_model_loading.py', 'Model Loading Tests'),
        ('tests/test_universal_dataset.py', 'Universal Dataset Tests'),
    ]

    results = []
    total_start = time.time()

    for test_file, description in tests:
        if not Path(test_file).exists():
            print(f"\n⚠️  SKIPPING: {test_file} not found")
            results.append((description, 'SKIPPED'))
            continue

        success = run_test(test_file, description)
        results.append((description, 'PASS' if success else 'FAIL'))

        if not success:
            print(f"\n⚠️  {description} failed, but continuing...")

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, status in results if status == 'PASS')
    failed = sum(1 for _, status in results if status == 'FAIL')
    skipped = sum(1 for _, status in results if status == 'SKIPPED')

    for i, (description, status) in enumerate(results, 1):
        if status == 'PASS':
            symbol = '✅'
        elif status == 'FAIL':
            symbol = '❌'
        else:
            symbol = '⚠️'

        print(f"{i}. {symbol} {description:40s} {status}")

    print("="*60)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Skipped: {skipped} ⚠️")
    print(f"Time: {total_elapsed:.2f}s")
    print("="*60)

    if failed == 0 and passed > 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n✅ System is ready for training!")
        print("\nNext steps:")
        print("  1. Validate notebook: python validate_notebook.py KAGGLE_COMPLETE.ipynb")
        print("  2. Upload to Kaggle")
        print("  3. Run smoke test (SMOKE_TEST = True)")
        print("  4. Verify F1 > 0.30")
        print("  5. Run full training (SMOKE_TEST = False)")
        return 0
    elif failed > 0:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nPlease fix failed tests before proceeding!")
        print("Check TROUBLESHOOTING_GUIDE.md for help")
        return 1
    else:
        print("\n⚠️  NO TESTS RAN")
        print("\nPlease check test files exist")
        return 1


if __name__ == '__main__':
    sys.exit(main())
