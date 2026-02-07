"""Download medical NLP datasets using HuggingFace datasets library.

Much easier and more reliable than manual downloads!

Usage:
    python scripts/download_datasets_hf.py --all
"""

import sys
import io
import argparse
from pathlib import Path

# Fix Windows encoding for Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("[!] 'datasets' library not installed")
    print("Install with: pip install datasets")


def download_bc5cdr(data_dir: Path):
    """Download BC5CDR from HuggingFace."""
    print("\n" + "=" * 60)
    print("Downloading BC5CDR (Chemical-Disease NER + Relations)")
    print("=" * 60)

    # Try multiple sources (working as of 2026-02-07)
    sources = [
        ("ghadeermobasher/BC5CDR-Chemical-Disease", None),  # Community dataset
        ("JHnlp/BioCreative-V-CDR-Corpus", None),  # Official GitHub mirror
    ]

    for source, config in sources:
        try:
            print(f"  Trying {source}...")
            dataset = load_dataset(source)

            # Save to disk
            output_dir = data_dir / "bc5cdr"
            dataset.save_to_disk(str(output_dir))

            splits = list(dataset.keys())
            print(f"[OK] BC5CDR downloaded from {source}")
            print(f"  Splits: {', '.join(splits)}")
            print(f"  Location: {output_dir}")
            return True
        except Exception as e:
            print(f"  [X] {source} failed: {str(e)[:100]}")
            continue

    print("[X] All BC5CDR sources failed")
    print("  Manual download from: https://github.com/JHnlp/BioCreative-V-CDR-Corpus")
    return False


def download_ncbi_disease(data_dir: Path):
    """Download NCBI-Disease from HuggingFace."""
    print("\n" + "=" * 60)
    print("Downloading NCBI-Disease (Disease NER)")
    print("=" * 60)

    sources = [
        ("ncbi/ncbi_disease", None),  # Official NCBI dataset
        ("ncbi_disease", None),  # Alternative name
    ]

    for source, config in sources:
        try:
            print(f"  Trying {source}...")
            dataset = load_dataset(source)

            output_dir = data_dir / "ncbi_disease"
            dataset.save_to_disk(str(output_dir))

            splits = list(dataset.keys())
            print(f"[OK] NCBI-Disease downloaded from {source}")
            print(f"  Splits: {', '.join(splits)}")
            print(f"  Location: {output_dir}")
            return True
        except Exception as e:
            print(f"  [X] {source} failed: {str(e)[:100]}")
            continue

    print("[X] All NCBI-Disease sources failed")
    print("  Manual download from: https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/")
    return False


def download_ddi(data_dir: Path):
    """Download DDI from HuggingFace."""
    print("\n" + "=" * 60)
    print("Downloading DDI (Drug-Drug Interactions)")
    print("=" * 60)

    print("  [!] DDI dataset not available in modern HuggingFace format")
    print("  Manual download required from: https://github.com/isegura/DDICorpus")
    print("  Or use: python scripts/download_datasets_manual.py --dataset ddi")
    return False


def download_gad(data_dir: Path):
    """Download GAD from HuggingFace."""
    print("\n" + "=" * 60)
    print("Downloading GAD (Gene-Disease Associations)")
    print("=" * 60)

    print("  [!] GAD dataset not available in modern HuggingFace format")
    print("  Note: GAD database was frozen as of 09/01/2014")
    print("  Manual download required - use BioBERT preprocessed version")
    print("  Or use: python scripts/download_datasets_manual.py --dataset gad")
    return False


def download_pubmedqa(data_dir: Path):
    """Download PubMedQA from HuggingFace."""
    print("\n" + "=" * 60)
    print("Downloading PubMedQA (Medical QA)")
    print("=" * 60)

    sources = [
        ("pubmed_qa", "pqa_labeled"),
        ("qiaojin/PubMedQA", "pqa_labeled"),
    ]

    for source, config in sources:
        try:
            print(f"  Trying {source}...")
            if config:
                dataset = load_dataset(source, config)
            else:
                dataset = load_dataset(source)

            output_dir = data_dir / "pubmedqa"
            dataset.save_to_disk(str(output_dir))

            splits = list(dataset.keys())
            print(f"[OK] PubMedQA downloaded from {source}")
            print(f"  Splits: {', '.join(splits)}")
            print(f"  Location: {output_dir}")
            return True
        except Exception as e:
            print(f"  [X] {source} failed: {str(e)[:100]}")
            continue

    print("[X] All PubMedQA sources failed")
    print("  Manual download from: https://github.com/pubmedqa/pubmedqa")
    return False


def main():
    parser = argparse.ArgumentParser(description="Download datasets via HuggingFace")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", type=str,
                        choices=["bc5cdr", "ncbi", "ddi", "gad", "pubmedqa"],
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
    print("HUGGINGFACE DATASETS DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {data_dir.absolute()}\n")

    results = {}

    if args.all:
        results['bc5cdr'] = download_bc5cdr(data_dir)
        results['ncbi_disease'] = download_ncbi_disease(data_dir)
        results['ddi'] = download_ddi(data_dir)
        results['gad'] = download_gad(data_dir)
        results['pubmedqa'] = download_pubmedqa(data_dir)
    elif args.dataset:
        if args.dataset == "bc5cdr":
            results['bc5cdr'] = download_bc5cdr(data_dir)
        elif args.dataset == "ncbi":
            results['ncbi_disease'] = download_ncbi_disease(data_dir)
        elif args.dataset == "ddi":
            results['ddi'] = download_ddi(data_dir)
        elif args.dataset == "gad":
            results['gad'] = download_gad(data_dir)
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
        print("\nNext steps:")
        print("  1. Implement parsers in src/data/")
        print("  2. Run: python verify_imports.py")
        print("  3. Start experiments!")
    else:
        print(f"\n[!] {total_count - success_count} dataset(s) failed")
        print("  Check errors above for details")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
