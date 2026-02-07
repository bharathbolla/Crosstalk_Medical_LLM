"""Download medical NLP datasets using Parquet files from HuggingFace.

HuggingFace automatically converts datasets to Parquet format on the
refs/convert/parquet branch. This script loads them directly from Parquet
to avoid deprecated loading scripts.

Usage:
    python scripts/download_datasets_parquet.py --all
    python scripts/download_datasets_parquet.py --dataset bc5cdr
"""

import sys
import io
import argparse
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("[!] 'datasets' library not installed")
    print("Install with: pip install datasets")


def download_bc5cdr(data_dir: Path):
    """Download BC5CDR from BLURB benchmark (Parquet files)."""
    print("\n" + "=" * 60)
    print("Downloading BC5CDR (Chemical-Disease NER)")
    print("=" * 60)

    try:
        base_url = "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet"

        # BC5CDR has two sub-datasets: bc5chem and bc5disease
        print("  Loading BC5 Chemical...")
        ds_chem = load_dataset('parquet', data_files={
            'train': f'{base_url}/bc5chem/train/0000.parquet',
            'validation': f'{base_url}/bc5chem/validation/0000.parquet',
            'test': f'{base_url}/bc5chem/test/0000.parquet'
        })

        print("  Loading BC5 Disease...")
        ds_disease = load_dataset('parquet', data_files={
            'train': f'{base_url}/bc5disease/train/0000.parquet',
            'validation': f'{base_url}/bc5disease/validation/0000.parquet',
            'test': f'{base_url}/bc5disease/test/0000.parquet'
        })

        # Save to disk
        output_dir = data_dir / "bc5cdr"
        output_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "chem").mkdir(exist_ok=True)
        (output_dir / "disease").mkdir(exist_ok=True)

        ds_chem.save_to_disk(str(output_dir / "chem"))
        ds_disease.save_to_disk(str(output_dir / "disease"))

        print(f"[OK] BC5CDR downloaded successfully")
        print(f"  Chem: {len(ds_chem['train'])} train, {len(ds_chem['validation'])} val, {len(ds_chem['test'])} test")
        print(f"  Disease: {len(ds_disease['train'])} train, {len(ds_disease['validation'])} val, {len(ds_disease['test'])} test")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] BC5CDR download failed: {str(e)[:200]}")
        return False


def download_ncbi_disease(data_dir: Path):
    """Download NCBI-Disease from BLURB benchmark (Parquet files)."""
    print("\n" + "=" * 60)
    print("Downloading NCBI-Disease (Disease NER)")
    print("=" * 60)

    try:
        base_url = "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/ncbi_disease"

        print("  Loading from Parquet files...")
        dataset = load_dataset('parquet', data_files={
            'train': f'{base_url}/train/0000.parquet',
            'validation': f'{base_url}/validation/0000.parquet',
            'test': f'{base_url}/test/0000.parquet'
        })

        # Save to disk
        output_dir = data_dir / "ncbi_disease"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] NCBI-Disease downloaded successfully")
        print(f"  Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] NCBI-Disease download failed: {str(e)[:200]}")
        return False


def download_bc2gm(data_dir: Path):
    """Download BC2GM (Gene/Protein NER) from BLURB (Parquet files)."""
    print("\n" + "=" * 60)
    print("Downloading BC2GM (Gene/Protein NER)")
    print("=" * 60)

    try:
        base_url = "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/bc2gm"

        print("  Loading from Parquet files...")
        dataset = load_dataset('parquet', data_files={
            'train': f'{base_url}/train/0000.parquet',
            'validation': f'{base_url}/validation/0000.parquet',
            'test': f'{base_url}/test/0000.parquet'
        })

        # Save to disk
        output_dir = data_dir / "bc2gm"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] BC2GM downloaded successfully")
        print(f"  Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] BC2GM download failed: {str(e)[:200]}")
        return False


def download_jnlpba(data_dir: Path):
    """Download JNLPBA (Bio-entity NER) from BLURB (Parquet files)."""
    print("\n" + "=" * 60)
    print("Downloading JNLPBA (Bio-entity NER)")
    print("=" * 60)

    try:
        base_url = "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/jnlpba"

        print("  Loading from Parquet files...")
        dataset = load_dataset('parquet', data_files={
            'train': f'{base_url}/train/0000.parquet',
            'validation': f'{base_url}/validation/0000.parquet',
            'test': f'{base_url}/test/0000.parquet'
        })

        # Save to disk
        output_dir = data_dir / "jnlpba"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] JNLPBA downloaded successfully")
        print(f"  Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] JNLPBA download failed: {str(e)[:200]}")
        return False


def download_pubmedqa(data_dir: Path):
    """Download PubMedQA (Medical QA)."""
    print("\n" + "=" * 60)
    print("Downloading PubMedQA (Medical QA)")
    print("=" * 60)

    try:
        print("  Loading PubMedQA...")
        dataset = load_dataset("pubmed_qa", "pqa_labeled")

        output_dir = data_dir / "pubmedqa"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] PubMedQA downloaded successfully")
        print(f"  Train: {len(dataset['train'])}")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] PubMedQA download failed: {str(e)[:200]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download medical datasets from Parquet files")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", type=str,
                        choices=["bc5cdr", "ncbi", "bc2gm", "jnlpba", "pubmedqa"],
                        help="Download specific dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Output directory")

    args = parser.parse_args()

    if not HAS_DATASETS:
        print("\nPlease install the datasets library:")
        print("  pip install datasets")
        return 1

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PARQUET-BASED DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {data_dir.absolute()}\n")
    print("NOTE: Loading from HuggingFace auto-converted Parquet files")
    print("      to avoid deprecated loading scripts.\n")

    results = {}

    if args.all:
        results['bc5cdr'] = download_bc5cdr(data_dir)
        results['ncbi_disease'] = download_ncbi_disease(data_dir)
        results['bc2gm'] = download_bc2gm(data_dir)
        results['jnlpba'] = download_jnlpba(data_dir)
        results['pubmedqa'] = download_pubmedqa(data_dir)
    elif args.dataset:
        if args.dataset == "bc5cdr":
            results['bc5cdr'] = download_bc5cdr(data_dir)
        elif args.dataset == "ncbi":
            results['ncbi_disease'] = download_ncbi_disease(data_dir)
        elif args.dataset == "bc2gm":
            results['bc2gm'] = download_bc2gm(data_dir)
        elif args.dataset == "jnlpba":
            results['jnlpba'] = download_jnlpba(data_dir)
        elif args.dataset == "pubmedqa":
            results['pubmedqa'] = download_pubmedqa(data_dir)
    else:
        print("Please specify --all or --dataset")
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    print(f"\n[OK] Downloaded: {success_count}/{total_count} datasets")

    for name, success in results.items():
        status = "[OK]" if success else "[X]"
        print(f"  {status} {name}")

    if success_count == total_count:
        print("\n[SUCCESS] All datasets downloaded successfully!")
        print("\nDataset Details:")
        print("  - bc5cdr: Chemical & Disease NER (from BLURB)")
        print("  - ncbi_disease: Disease NER (from BLURB)")
        print("  - bc2gm: Gene/Protein NER (from BLURB)")
        print("  - jnlpba: Bio-entity NER (from BLURB)")
        print("  - pubmedqa: Medical Question Answering")
        print("\nNext steps:")
        print("  1. Implement parsers in src/data/")
        print("  2. Run: python verify_imports.py")
        print("  3. Start experiments!")
    else:
        print(f"\n[!] {total_count - success_count} dataset(s) failed")
        print("  Check errors above for details")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
