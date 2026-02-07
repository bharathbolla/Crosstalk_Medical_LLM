"""Download datasets using Parquet URLs (ChemProt excluded due to loading issues)."""

from datasets import load_dataset
from pathlib import Path

# Create data directory
data_path = Path("data/raw")
data_path.mkdir(parents=True, exist_ok=True)

print("📥 Downloading 7 datasets using Parquet method...\n")

# Parquet URL configurations (ChemProt excluded)
datasets_config = {
    "bc2gm": {
        "url": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/bc2gm",
        "splits": ["train", "validation", "test"]
    },
    "jnlpba": {
        "url": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/jnlpba",
        "splits": ["train", "validation", "test"]
    },
    # ChemProt excluded - causes FileNotFoundError with Parquet URLs
    # "chemprot": {
    #     "url": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/chemprot",
    #     "splits": ["train", "validation", "test"]
    # },
    "ddi": {
        "url": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/ddi_corpus",
        "splits": ["train", "test"]
    },
    "gad": {
        "url": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/gad",
        "splits": ["train", "test"]
    },
    "hoc": {
        "url": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/hallmarks_of_cancer",
        "splits": ["train", "validation", "test"]
    },
    "pubmedqa": {
        "url": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/pubmed_qa",
        "splits": ["train", "validation", "test"]
    },
    "biosses": {
        "url": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/biosses",
        "splits": ["train", "validation", "test"]
    }
}

total_samples = 0
successful = 0

for name, config in datasets_config.items():
    print(f"📦 Downloading {name}...")
    base_url = config["url"]

    try:
        # Build data_files dict
        data_files = {}
        for split in config["splits"]:
            data_files[split] = f"{base_url}/{split}/0000.parquet"

        # Load using Parquet method
        dataset = load_dataset("parquet", data_files=data_files)
        dataset.save_to_disk(str(data_path / name))

        # Show stats
        train_size = len(dataset["train"])
        total_samples += train_size
        successful += 1

        print(f"  ✓ {train_size:,} training samples")
        print(f"  ✓ Saved to: {data_path / name}\n")

    except Exception as e:
        print(f"  ✗ ERROR: {str(e)[:100]}\n")
        continue

print(f"\n{'='*60}")
print(f"✅ Successfully downloaded {successful}/7 datasets")
print(f"📊 Total training samples: {total_samples:,}")
print(f"{'='*60}")
print(f"\nNote: ChemProt excluded due to Parquet URL compatibility issues.")
print(f"You still have 7 diverse datasets for multi-task learning!")
