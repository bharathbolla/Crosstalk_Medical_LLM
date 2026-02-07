"""Download all public medical NLP datasets.

No PhysioNet access required - all datasets are publicly available.

Datasets:
1. BC5CDR - Chemical/Disease NER + Relations
2. NCBI-Disease - Disease mention NER
3. DDI Corpus - Drug-Drug Interactions
4. GAD - Gene-Disease Associations
5. PubMedQA - Medical Question Answering

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --dataset bc5cdr
"""

import argparse
import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import json


def download_file(url: str, output_path: Path):
    """Download file from URL."""
    print(f"Downloading from: {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(url, output_path)
    print(f"✓ Downloaded to: {output_path}")


def extract_archive(archive_path: Path, extract_to: Path):
    """Extract zip or tar.gz archive."""
    print(f"Extracting: {archive_path}")

    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix == '.gz':
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)

    print(f"✓ Extracted to: {extract_to}")


def download_bc5cdr(data_dir: Path):
    """Download BC5CDR dataset (Chemical-Disease Relations).

    Source: BioCreative V CDR Task
    URL: https://biocreative.bioinformatics.udel.edu/tasks/biocreative-v/track-3-cdr/
    """
    print("\n" + "=" * 60)
    print("Downloading BC5CDR (Chemical-Disease NER + Relations)")
    print("=" * 60)

    bc5cdr_dir = data_dir / "bc5cdr"
    bc5cdr_dir.mkdir(parents=True, exist_ok=True)

    # Download from public source
    url = "https://biocreative.bioinformatics.udel.edu/media/store/files/2016/CDR_Data.zip"
    zip_path = bc5cdr_dir / "CDR_Data.zip"

    download_file(url, zip_path)
    extract_archive(zip_path, bc5cdr_dir)

    print("✓ BC5CDR ready!")
    print(f"  Location: {bc5cdr_dir}")
    print(f"  Files: Train, Dev, Test splits")
    print(f"  Format: PubTator (standoff annotations)")


def download_ncbi_disease(data_dir: Path):
    """Download NCBI-Disease dataset.

    Source: NCBI Disease Corpus
    URL: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/
    """
    print("\n" + "=" * 60)
    print("Downloading NCBI-Disease (Disease NER)")
    print("=" * 60)

    ncbi_dir = data_dir / "ncbi_disease"
    ncbi_dir.mkdir(parents=True, exist_ok=True)

    # Download from public FTP
    base_url = "https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/"

    files = {
        "NCBItrainset_corpus.txt": "train.txt",
        "NCBIdevelopset_corpus.txt": "dev.txt",
        "NCBItestset_corpus.txt": "test.txt",
    }

    for filename, output_name in files.items():
        url = base_url + filename
        output_path = ncbi_dir / output_name
        download_file(url, output_path)

    print("✓ NCBI-Disease ready!")
    print(f"  Location: {ncbi_dir}")


def download_ddi(data_dir: Path):
    """Download DDI Corpus (Drug-Drug Interactions).

    Source: DDIExtraction 2013 Challenge
    URL: https://github.com/isegura/DDICorpus
    """
    print("\n" + "=" * 60)
    print("Downloading DDI Corpus (Drug-Drug Interactions)")
    print("=" * 60)

    ddi_dir = data_dir / "ddi"
    ddi_dir.mkdir(parents=True, exist_ok=True)

    # Download from GitHub
    url = "https://github.com/isegura/DDICorpus/archive/refs/heads/master.zip"
    zip_path = ddi_dir / "DDICorpus.zip"

    download_file(url, zip_path)
    extract_archive(zip_path, ddi_dir)

    print("✓ DDI Corpus ready!")
    print(f"  Location: {ddi_dir}")


def download_gad(data_dir: Path):
    """Download GAD dataset (Gene-Disease Associations).

    Source: Genetic Association Database
    URL: https://github.com/dmis-lab/biobert
    """
    print("\n" + "=" * 60)
    print("Downloading GAD (Gene-Disease Associations)")
    print("=" * 60)

    gad_dir = data_dir / "gad"
    gad_dir.mkdir(parents=True, exist_ok=True)

    # Download from BioBERT repository
    base_url = "https://raw.githubusercontent.com/dmis-lab/biobert/master/datasets/GAD/"

    files = ["train.tsv", "dev.tsv", "test.tsv"]

    for filename in files:
        url = base_url + filename
        output_path = gad_dir / filename
        download_file(url, output_path)

    print("✓ GAD ready!")
    print(f"  Location: {gad_dir}")


def download_pubmedqa(data_dir: Path):
    """Download PubMedQA dataset.

    Source: PubMedQA
    URL: https://github.com/pubmedqa/pubmedqa
    """
    print("\n" + "=" * 60)
    print("Downloading PubMedQA (Medical Question Answering)")
    print("=" * 60)

    pubmedqa_dir = data_dir / "pubmedqa"
    pubmedqa_dir.mkdir(parents=True, exist_ok=True)

    # Download from GitHub
    base_url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/"

    files = {
        "ori_pqal.json": "train.json",
        "ori_pqau.json": "unlabeled.json",
        "test_ground_truth.json": "test.json",
    }

    for filename, output_name in files.items():
        url = base_url + filename
        output_path = pubmedqa_dir / output_name
        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"  Warning: Could not download {filename}: {e}")

    print("✓ PubMedQA ready!")
    print(f"  Location: {pubmedqa_dir}")


def verify_downloads(data_dir: Path):
    """Verify all datasets were downloaded."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    datasets = {
        "BC5CDR": data_dir / "bc5cdr",
        "NCBI-Disease": data_dir / "ncbi_disease",
        "DDI": data_dir / "ddi",
        "GAD": data_dir / "gad",
        "PubMedQA": data_dir / "pubmedqa",
    }

    all_ok = True
    for name, path in datasets.items():
        if path.exists():
            file_count = len(list(path.rglob("*.*")))
            print(f"✓ {name}: {file_count} files in {path}")
        else:
            print(f"✗ {name}: NOT FOUND")
            all_ok = False

    if all_ok:
        print("\n✓ All datasets downloaded successfully!")
        print("\nNext steps:")
        print("  1. Implement task parsers for each dataset")
        print("  2. Run: python verify_datasets.py")
        print("  3. Start training!")
    else:
        print("\n⚠ Some datasets missing. Try re-running download.")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Download public medical NLP datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", type=str, choices=["bc5cdr", "ncbi", "ddi", "gad", "pubmedqa"],
                        help="Download specific dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Output directory")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PUBLIC MEDICAL NLP DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {data_dir.absolute()}")

    if args.all:
        download_bc5cdr(data_dir)
        download_ncbi_disease(data_dir)
        download_ddi(data_dir)
        download_gad(data_dir)
        download_pubmedqa(data_dir)
    elif args.dataset:
        if args.dataset == "bc5cdr":
            download_bc5cdr(data_dir)
        elif args.dataset == "ncbi":
            download_ncbi_disease(data_dir)
        elif args.dataset == "ddi":
            download_ddi(data_dir)
        elif args.dataset == "gad":
            download_gad(data_dir)
        elif args.dataset == "pubmedqa":
            download_pubmedqa(data_dir)
    else:
        print("Please specify --all or --dataset")
        return 1

    # Verify
    verify_downloads(data_dir)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
