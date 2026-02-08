"""
Test data loading for all 8 datasets.
Run this BEFORE starting any experiment!
"""

import sys
import io
import pickle
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_all_datasets_exist():
    """Test all pickle files exist."""
    print("\n" + "="*60)
    print("TEST 1: All Datasets Exist")
    print("="*60)

    datasets = ['bc2gm', 'jnlpba', 'chemprot', 'ddi', 'gad', 'hoc', 'pubmedqa', 'biosses']

    all_exist = True
    for ds in datasets:
        pkl_file = Path('data/pickle') / f'{ds}.pkl'
        if pkl_file.exists():
            size_mb = pkl_file.stat().st_size / 1_000_000
            print(f"✅ {ds:12s} | {size_mb:.1f} MB")
        else:
            print(f"❌ {ds:12s} | NOT FOUND at {pkl_file}")
            all_exist = False

    assert all_exist, "Some datasets missing!"
    print("\n✅ All datasets exist")
    return True


def test_all_datasets_have_train():
    """Test all datasets have train split."""
    print("\n" + "="*60)
    print("TEST 2: All Datasets Have Train Split")
    print("="*60)

    datasets = ['bc2gm', 'jnlpba', 'chemprot', 'ddi', 'gad', 'hoc', 'pubmedqa', 'biosses']

    all_good = True
    for ds in datasets:
        pkl_file = Path('data/pickle') / f'{ds}.pkl'
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)

            if 'train' not in data:
                print(f"❌ {ds:12s} | Missing 'train' split")
                all_good = False
                continue

            if len(data['train']) == 0:
                print(f"❌ {ds:12s} | Empty 'train' split")
                all_good = False
                continue

            # Check for validation or test
            val_split = 'validation' if 'validation' in data else 'test'
            val_size = len(data.get(val_split, []))

            print(f"✅ {ds:12s} | Train: {len(data['train']):5,} | Val/Test: {val_size:4,}")

        except Exception as e:
            print(f"❌ {ds:12s} | Error: {e}")
            all_good = False

    assert all_good, "Some datasets have issues!"
    print("\n✅ All datasets have train split")
    return True


def test_sample_structure():
    """Test first sample structure from BC2GM."""
    print("\n" + "="*60)
    print("TEST 3: Sample Structure (BC2GM)")
    print("="*60)

    pkl_file = Path('data/pickle/bc2gm.pkl')
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    sample = data['train'][0]
    print(f"Sample keys: {list(sample.keys())}")

    # Check required fields
    required = ['tokens', 'labels']
    for field in required:
        assert field in sample, f"Missing required field: {field}"
        print(f"✅ Has '{field}': {type(sample[field])}")

    # Show example
    print(f"\nExample:")
    print(f"  Tokens: {sample['tokens'][:5]}...")
    print(f"  Labels: {sample['labels'][:5]}...")

    print("\n✅ Sample structure correct")
    return True


if __name__ == '__main__':
    print("="*60)
    print("DATA LOADING TESTS")
    print("="*60)

    try:
        test_all_datasets_exist()
        test_all_datasets_have_train()
        test_sample_structure()

        print("\n" + "="*60)
        print("✅ ALL DATA LOADING TESTS PASSED")
        print("="*60)
        print("\nReady to proceed with training!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
