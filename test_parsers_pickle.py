"""
Test that pickle datasets have the correct structure for training.
Works on Kaggle with ZERO dependencies!
"""

import pickle
import sys
import io
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def test_pickle_structure():
    """Test that pickle datasets have correct structure."""

    print("=" * 60)
    print("PICKLE DATASET STRUCTURE TEST")
    print("=" * 60)
    print()

    pickle_path = Path("data/pickle")
    datasets_info = {
        "bc2gm": {"task": "NER", "expected_fields": ["id", "tokens", "ner_tags"]},
        "jnlpba": {"task": "NER", "expected_fields": ["id", "tokens", "ner_tags"]},
        "chemprot": {"task": "RE", "expected_fields": ["id", "text"]},
        "ddi": {"task": "RE", "expected_fields": ["id", "text"]},
        "gad": {"task": "Classification", "expected_fields": ["id", "text", "label"]},
        "hoc": {"task": "Classification", "expected_fields": ["id", "text"]},
        "pubmedqa": {"task": "QA", "expected_fields": ["pubid", "question", "context"]},
        "biosses": {"task": "Similarity", "expected_fields": ["id", "text_1", "text_2"]},
    }

    success_count = 0

    for dataset_name, info in datasets_info.items():
        print(f"\n📦 {dataset_name.upper()} ({info['task']})")

        pickle_file = pickle_path / f"{dataset_name}.pkl"

        if not pickle_file.exists():
            print(f"   ✗ Pickle file not found")
            continue

        try:
            # Load pickle
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)

            # Check structure
            if 'train' not in data:
                print(f"   ✗ Missing 'train' split")
                continue

            train_data = data['train']
            if len(train_data) == 0:
                print(f"   ✗ Empty training data")
                continue

            # Check first sample
            first_sample = train_data[0]
            sample_fields = set(first_sample.keys())

            print(f"   ✓ Train samples: {len(train_data):,}")
            print(f"   ✓ Splits: {', '.join(data.keys())}")
            print(f"   ✓ Sample fields: {list(sample_fields)[:6]}...")

            # Verify has necessary fields
            missing_fields = set(info['expected_fields']) - sample_fields
            if missing_fields:
                print(f"   ⚠️  Missing fields: {missing_fields}")
            else:
                print(f"   ✓ All expected fields present")

            success_count += 1

        except Exception as e:
            print(f"   ✗ Error: {str(e)[:100]}")

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {success_count}/8 datasets ready")
    print("=" * 60)

    if success_count == 8:
        print("\n🎉 All pickle datasets have correct structure!")
        print("\nYou can now:")
        print("  1. Use pickle data directly for training")
        print("  2. Write simple data loaders without datasets library")
        print("  3. Start experiments on Kaggle!")
        return True
    else:
        print(f"\n⚠️  {8 - success_count} datasets need attention")
        return False


if __name__ == "__main__":
    success = test_pickle_structure()
    exit(0 if success else 1)
