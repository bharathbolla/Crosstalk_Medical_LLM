"""
Convert datasets from Arrow format to simple pickle format
Run this LOCALLY where datasets library works
Then commit the pickle files for Kaggle to use
"""

import sys
import io
# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pickle
from pathlib import Path
from datasets import load_from_disk

def convert_datasets():
    """Convert all datasets to pickle format."""

    data_path = Path("data/raw")
    output_path = Path("data/pickle")
    output_path.mkdir(parents=True, exist_ok=True)

    datasets_list = ["bc2gm", "jnlpba", "chemprot", "ddi", "gad", "hoc", "pubmedqa", "biosses"]

    print("Converting datasets to pickle format...")
    print("=" * 60)

    for dataset_name in datasets_list:
        print(f"\n📦 {dataset_name.upper()}")

        dataset_dir = data_path / dataset_name

        try:
            # Load the full dataset
            dataset = load_from_disk(str(dataset_dir))

            # Convert each split to simple dict format
            simple_data = {}

            for split_name in dataset.keys():
                split_data = dataset[split_name]

                # Convert to list of dicts (simple format)
                simple_data[split_name] = [
                    {k: v for k, v in example.items()}
                    for example in split_data
                ]

                print(f"   ✓ {split_name}: {len(simple_data[split_name])} samples")

            # Save as pickle
            output_file = output_path / f"{dataset_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(simple_data, f)

            print(f"   ✓ Saved to {output_file}")

        except Exception as e:
            print(f"   ✗ Error: {str(e)}")

    print("\n" + "=" * 60)
    print("✅ Conversion complete!")
    print(f"\nPickle files saved in: {output_path}")
    print("\nNow commit these files and use them on Kaggle!")

if __name__ == "__main__":
    convert_datasets()
