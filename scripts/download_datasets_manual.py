"""Manual dataset downloads for medical NLP datasets.

These datasets use deprecated HuggingFace loading scripts,
so we download them directly from original sources.

Usage:
    python scripts/download_datasets_manual.py --all
    python scripts/download_datasets_manual.py --dataset bc5cdr
"""

import sys
import io
import argparse
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def download_file(url: str, output_path: Path, description: str = "file"):
    """Download a file with progress."""
    try:
        print(f"  Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"  [OK] Downloaded {description} ({file_size:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [X] Download failed: {str(e)[:100]}")
        return False


def extract_archive(archive_path: Path, output_dir: Path):
    """Extract zip or tar.gz archive."""
    try:
        print(f"  Extracting to {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
        elif archive_path.suffix == '.gz' or archive_path.suffix == '.tgz':
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(output_dir)
        else:
            print(f"  [X] Unknown archive format: {archive_path.suffix}")
            return False

        print(f"  [OK] Extracted successfully")
        return True
    except Exception as e:
        print(f"  [X] Extraction failed: {str(e)[:100]}")
        return False


def download_bc5cdr(data_dir: Path):
    """Download BC5CDR from BioCreative."""
    print("\n" + "=" * 60)
    print("Downloading BC5CDR (Chemical-Disease NER + Relations)")
    print("=" * 60)

    output_dir = data_dir / "bc5cdr"
    output_dir.mkdir(parents=True, exist_ok=True)

    # BC5CDR is available from multiple sources
    sources = [
        {
            "name": "PubTator Central",
            "url": "https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/CDR_Data.zip",
            "format": "PubTator"
        },
    ]

    for source in sources:
        print(f"\nTrying {source['name']}...")
        archive_path = output_dir / "BC5CDR.zip"

        if download_file(source['url'], archive_path, "BC5CDR dataset"):
            if extract_archive(archive_path, output_dir):
                # Clean up archive
                archive_path.unlink()

                # Count files
                files = list(output_dir.rglob("*.txt")) + list(output_dir.rglob("*.tsv"))
                print(f"\n[OK] BC5CDR downloaded successfully")
                print(f"  Format: {source['format']}")
                print(f"  Files: {len(files)} annotation files")
                print(f"  Location: {output_dir}")
                return True

    print("\n[X] BC5CDR download failed from all sources")
    print("  Alternative: Download manually from https://biocreative.bioinformatics.udel.edu/")
    return False


def download_ncbi_disease(data_dir: Path):
    """Download NCBI-Disease corpus."""
    print("\n" + "=" * 60)
    print("Downloading NCBI-Disease (Disease NER)")
    print("=" * 60)

    output_dir = data_dir / "ncbi_disease"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "name": "NCBI FTP",
            "url": "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBI_corpus.zip",
            "format": "BioC XML"
        },
    ]

    for source in sources:
        print(f"\nTrying {source['name']}...")
        archive_path = output_dir / "NCBI_corpus.zip"

        if download_file(source['url'], archive_path, "NCBI-Disease corpus"):
            if extract_archive(archive_path, output_dir):
                archive_path.unlink()

                files = list(output_dir.rglob("*.txt")) + list(output_dir.rglob("*.xml"))
                print(f"\n[OK] NCBI-Disease downloaded successfully")
                print(f"  Format: {source['format']}")
                print(f"  Files: {len(files)} annotation files")
                print(f"  Location: {output_dir}")
                return True

    print("\n[X] NCBI-Disease download failed")
    print("  Alternative: Download from https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/")
    return False


def download_ddi(data_dir: Path):
    """Download DDI Corpus (Drug-Drug Interactions)."""
    print("\n" + "=" * 60)
    print("Downloading DDI (Drug-Drug Interactions)")
    print("=" * 60)

    output_dir = data_dir / "ddi"
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "name": "GitHub Mirror",
            "url": "https://github.com/isegura/DDICorpus/archive/refs/heads/master.zip",
            "format": "XML"
        },
    ]

    for source in sources:
        print(f"\nTrying {source['name']}...")
        archive_path = output_dir / "DDI.zip"

        if download_file(source['url'], archive_path, "DDI Corpus"):
            if extract_archive(archive_path, output_dir):
                archive_path.unlink()

                files = list(output_dir.rglob("*.xml"))
                print(f"\n[OK] DDI Corpus downloaded successfully")
                print(f"  Format: {source['format']}")
                print(f"  Files: {len(files)} XML files")
                print(f"  Location: {output_dir}")
                return True

    print("\n[X] DDI download failed")
    print("  Alternative: Download from https://github.com/isegura/DDICorpus")
    return False


def download_gad(data_dir: Path):
    """Download GAD (Gene-Disease Associations)."""
    print("\n" + "=" * 60)
    print("Downloading GAD (Gene-Disease Associations)")
    print("=" * 60)

    output_dir = data_dir / "gad"
    output_dir.mkdir(parents=True, exist_ok=True)

    # GAD is distributed as part of BioBERT datasets
    sources = [
        {
            "name": "BioBERT GitHub",
            "url": "https://github.com/dmis-lab/biobert/raw/master/dataset/GAD/train.tsv",
            "file": "train.tsv"
        },
        {
            "name": "BioBERT GitHub",
            "url": "https://github.com/dmis-lab/biobert/raw/master/dataset/GAD/dev.tsv",
            "file": "dev.tsv"
        },
        {
            "name": "BioBERT GitHub",
            "url": "https://github.com/dmis-lab/biobert/raw/master/dataset/GAD/test.tsv",
            "file": "test.tsv"
        },
    ]

    success_count = 0
    for source in sources:
        file_path = output_dir / source['file']
        if download_file(source['url'], file_path, source['file']):
            success_count += 1

    if success_count == len(sources):
        print(f"\n[OK] GAD downloaded successfully")
        print(f"  Format: TSV (tab-separated)")
        print(f"  Splits: train, dev, test")
        print(f"  Location: {output_dir}")
        return True
    else:
        print(f"\n[X] GAD download incomplete ({success_count}/{len(sources)} files)")
        print("  Alternative: Download from https://github.com/dmis-lab/biobert")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download medical datasets manually")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", type=str,
                        choices=["bc5cdr", "ncbi", "ddi", "gad"],
                        help="Download specific dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Output directory")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MANUAL MEDICAL DATASETS DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {data_dir.absolute()}\n")

    results = {}

    if args.all:
        results['bc5cdr'] = download_bc5cdr(data_dir)
        results['ncbi_disease'] = download_ncbi_disease(data_dir)
        results['ddi'] = download_ddi(data_dir)
        results['gad'] = download_gad(data_dir)
    elif args.dataset:
        if args.dataset == "bc5cdr":
            results['bc5cdr'] = download_bc5cdr(data_dir)
        elif args.dataset == "ncbi":
            results['ncbi_disease'] = download_ncbi_disease(data_dir)
        elif args.dataset == "ddi":
            results['ddi'] = download_ddi(data_dir)
        elif args.dataset == "gad":
            results['gad'] = download_gad(data_dir)
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
        print("\nNote: These are in original formats (PubTator, BioC, XML, TSV)")
        print("Next steps:")
        print("  1. Implement parsers in src/data/ to convert to UnifiedSample format")
        print("  2. Run: python verify_imports.py")
        print("  3. Start experiments!")
    else:
        print(f"\n[!] {total_count - success_count} dataset(s) failed")
        print("  Check errors above for manual download instructions")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
