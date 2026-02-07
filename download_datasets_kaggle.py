"""Download all 8 datasets on Kaggle using the bigbio/blurb dataset."""

from datasets import load_dataset
from pathlib import Path

# Create data directory
data_path = Path("data/raw")
data_path.mkdir(parents=True, exist_ok=True)

print("Downloading all 8 datasets from bigbio/blurb...\n")

# Dataset configurations - use the bigbio/blurb repository
datasets_config = {
    "bc2gm": "bc2gm",
    "jnlpba": "jnlpba", 
    "chemprot": "chemprot",
    "ddi": "ddi_corpus",
    "gad": "gad",
    "hoc": "hallmarks_of_cancer",
    "pubmedqa": "pubmed_qa",
    "biosses": "biosses"
}

total_samples = 0
for name, subset in datasets_config.items():
    print(f"Downloading {name} (subset: {subset})...")
    
    try:
        # Load directly from HuggingFace using the subset name
        dataset = load_dataset("bigbio/blurb", subset, trust_remote_code=True)
        
        # Save to disk
        dataset.save_to_disk(str(data_path / name))
        
        # Show stats
        train_size = len(dataset["train"])
        total_samples += train_size
        print(f"  ✓ {name}: {train_size:,} training samples")
        
        # Show available splits
        splits = list(dataset.keys())
        print(f"  Available splits: {splits}\n")
        
    except Exception as e:
        print(f"  ✗ Error downloading {name}: {str(e)}\n")
        continue

print(f"\n{'='*60}")
print(f"✅ Download complete!")
print(f"📊 Total training samples: {total_samples:,}")
print(f"{'='*60}")
