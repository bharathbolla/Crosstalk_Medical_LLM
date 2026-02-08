"""
Test loading datasets from pickle files
ZERO external dependencies - just Python stdlib!
"""

import pickle
from pathlib import Path

def test_pickle_datasets():
    """Test loading pickle datasets - works everywhere!"""

    print("=" * 60)
    print("TESTING PICKLE DATASET LOADING")
    print("=" * 60)
    print("\nNO external dependencies needed!")
    print("Just Python standard library pickle module.\n")

    pickle_path = Path("data/pickle")
    datasets_list = ["bc2gm", "jnlpba", "chemprot", "ddi", "gad", "hoc", "pubmedqa", "biosses"]

    success = 0
    total_samples = 0

    for dataset_name in datasets_list:
        print(f"\n📦 {dataset_name.upper()}")

        pickle_file = pickle_path / f"{dataset_name}.pkl"

        if not pickle_file.exists():
            print(f"   ✗ Not found at {pickle_file}")
            continue

        try:
            # Load pickle file (no dependencies!)
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)

            # Show statistics
            splits = list(data.keys())
            train_samples = len(data.get('train', []))
            total_samples += train_samples

            print(f"   ✓ Loaded successfully!")
            print(f"   ✓ Splits: {', '.join(splits)}")
            print(f"   ✓ Train samples: {train_samples:,}")

            # Show first sample
            if train_samples > 0:
                first_sample = data['train'][0]
                sample_keys = list(first_sample.keys())
                print(f"   ✓ Sample fields: {sample_keys[:5]}...")

            success += 1

        except Exception as e:
            print(f"   ✗ Error: {str(e)}")

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Successfully loaded: {success}/8 datasets")
    print(f"📊 Total training samples: {total_samples:,}")
    print("=" * 60)

    if success == 8:
        print("\n🎉 ALL DATASETS WORK!")
        print("\nNo datasets library needed!")
        print("No pyarrow needed!")
        print("No version conflicts!")
        print("\nJust clone and use! ✅")
        return True
    else:
        print(f"\n⚠️  {8 - success} datasets failed")
        return False

if __name__ == "__main__":
    success = test_pickle_datasets()
    exit(0 if success else 1)
