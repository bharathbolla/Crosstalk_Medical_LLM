"""
Test loading datasets directly from Arrow files
NO datasets library dependency - just pyarrow
"""

import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def test_direct_load():
    """Test loading Arrow files directly."""

    print("=" * 60)
    print("TESTING DIRECT ARROW FILE LOADING")
    print("=" * 60)
    print("\nThis bypasses the datasets library completely!")
    print()

    data_path = Path("data/raw")
    datasets_to_test = ["bc2gm", "jnlpba", "chemprot", "ddi", "gad", "hoc", "pubmedqa", "biosses"]

    success = 0

    for dataset_name in datasets_to_test:
        print(f"\n📦 {dataset_name.upper()}")

        # Check if dataset folder exists
        dataset_dir = data_path / dataset_name / "train"

        if not dataset_dir.exists():
            print(f"   ✗ Not found at {dataset_dir}")
            continue

        # Find arrow files
        arrow_files = list(dataset_dir.glob("*.arrow"))

        if not arrow_files:
            print(f"   ✗ No .arrow files found")
            continue

        # Load the arrow file
        try:
            arrow_file = arrow_files[0]

            # Open with pyarrow
            with pa.memory_map(str(arrow_file), 'r') as source:
                table = pa.ipc.open_file(source).read_all()

            num_rows = len(table)
            columns = table.column_names

            print(f"   ✓ Loaded {num_rows:,} samples")
            print(f"   ✓ Columns: {columns[:5]}...")  # Show first 5 columns

            success += 1

        except Exception as e:
            print(f"   ✗ Error: {str(e)[:100]}")

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Successfully loaded: {success}/8 datasets")
    print("=" * 60)

    if success == 8:
        print("\n🎉 All datasets can be loaded!")
        print("The data is accessible - we just need working parsers.")
        return True
    else:
        print(f"\n⚠️  {8 - success} datasets failed to load")
        return False

if __name__ == "__main__":
    success = test_direct_load()
    exit(0 if success else 1)
