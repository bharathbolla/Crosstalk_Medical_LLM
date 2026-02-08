"""Download BLURB datasets using official HuggingFace method (WORKING VERSION).

This script loads each dataset individually from bigbio collection.
Tested and working as of 2026-02-07.
"""

import sys
import io
# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from datasets import load_dataset
from pathlib import Path

# Create data directory
data_path = Path("data/raw")
data_path.mkdir(parents=True, exist_ok=True)

print("📥 Downloading 7 medical NLP datasets from bigbio collection...\n")
print("=" * 60)

# Dataset configurations - each dataset has its own repo and config name
datasets_config = {
    "bc2gm": {
        "repo": "bigbio/blurb",
        "config": "bc2gm",
        "description": "Gene/protein NER from PubMed abstracts"
    },
    "jnlpba": {
        "repo": "bigbio/blurb",
        "config": "jnlpba",
        "description": "Bio-entity NER (protein, DNA, RNA, cell line, cell type)"
    },
    "ddi": {
        "repo": "bigbio/ddi_corpus",
        "config": "ddi_corpus_source",
        "description": "Drug-drug interaction extraction"
    },
    "gad": {
        "repo": "bigbio/gad",
        "config": "gad_blurb_bigbio_text",
        "description": "Gene-disease association classification"
    },
    "hoc": {
        "repo": "bigbio/hallmarks_of_cancer",
        "config": "hallmarks_of_cancer_source",
        "description": "Cancer hallmarks classification (multi-label)"
    },
    "pubmedqa": {
        "repo": "bigbio/pubmed_qa",
        "config": "pubmed_qa_labeled_fold0_source",
        "description": "Medical question answering"
    },
    "biosses": {
        "repo": "bigbio/biosses",
        "config": "biosses_bigbio_pairs",
        "description": "Biomedical sentence similarity"
    }
}

total_samples = 0
successful = 0
failed = []

for name, config in datasets_config.items():
    print(f"\n📦 {name.upper()}")
    print(f"   {config['description']}")
    print(f"   Repo: {config['repo']}")
    print(f"   Config: {config['config']}")

    try:
        # Load dataset with specific config
        dataset = load_dataset(
            config["repo"],
            name=config["config"],
            trust_remote_code=False  # Use built-in loader, not custom scripts
        )

        # Save to disk
        dataset.save_to_disk(str(data_path / name))

        # Show stats
        train_size = len(dataset["train"])
        total_samples += train_size
        successful += 1

        # Show split info
        splits_info = " + ".join([f"{split}: {len(dataset[split])}" for split in dataset.keys()])
        print(f"   ✓ Downloaded successfully!")
        print(f"   ✓ Splits: {splits_info}")
        print(f"   ✓ Saved to: {data_path / name}")

    except Exception as e:
        failed.append(name)
        print(f"   ✗ ERROR: {str(e)[:150]}")
        continue

# Summary
print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)
print(f"\n✅ Successfully downloaded: {successful}/7 datasets")
print(f"📊 Total training samples: {total_samples:,}")

if successful == 7:
    print("\n🎉 All datasets downloaded successfully!")
    print(f"\nDatasets saved in: {data_path.absolute()}")
    print("\nNext steps:")
    print("  1. Run: python test_parsers.py")
    print("  2. Run smoke test")
    print("  3. Start experiments")
else:
    print(f"\n⚠️  Failed datasets: {', '.join(failed)}")
    print("\nTry installing dependencies:")
    print("  pip install datasets")
    sys.exit(1)

print("\n" + "=" * 60)
