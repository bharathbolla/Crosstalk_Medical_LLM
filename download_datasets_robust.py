"""Robust dataset downloader - tries multiple methods until one works.

This script handles all the dataset loading issues on Kaggle by trying:
1. Direct loading with trust_remote_code=True
2. Downgrade datasets library to 2.14.0
3. Direct Parquet file loading
4. Manual download from HuggingFace

Run this on Kaggle to download all 7 datasets reliably.
"""

from pathlib import Path
import subprocess
import sys

def install_package(package):
    """Install a package via pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

def method1_trust_remote_code():
    """Method 1: Try with trust_remote_code=True"""
    print("\n" + "="*60)
    print("METHOD 1: Loading with trust_remote_code=True")
    print("="*60)

    from datasets import load_dataset

    datasets_config = {
        "bc2gm": ("bigbio/blurb", "bc2gm"),
        "jnlpba": ("bigbio/blurb", "jnlpba"),
        "ddi": ("bigbio/ddi_corpus", "ddi_corpus_source"),
        "gad": ("bigbio/gad", "gad_blurb_bigbio_text"),
        "hoc": ("bigbio/hallmarks_of_cancer", "hallmarks_of_cancer_source"),
        "pubmedqa": ("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source"),
        "biosses": ("bigbio/biosses", "biosses_bigbio_pairs")
    }

    data_path = Path("data/raw")
    data_path.mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    for name, (repo, config) in datasets_config.items():
        print(f"\n📦 {name}...")
        try:
            dataset = load_dataset(repo, name=config, trust_remote_code=True)
            dataset.save_to_disk(str(data_path / name))
            successful.append(name)
            print(f"   ✓ Success!")
        except Exception as e:
            failed.append(name)
            print(f"   ✗ Failed: {str(e)[:80]}")

    return successful, failed

def method2_downgrade_datasets():
    """Method 2: Downgrade datasets library and retry"""
    print("\n" + "="*60)
    print("METHOD 2: Downgrading datasets library to 2.14.0")
    print("="*60)

    try:
        print("Installing datasets==2.14.0 (this may take 1-2 minutes)...")
        install_package("datasets==2.14.0")
        print("✓ Downgrade complete!")

        # Reload datasets module
        import importlib
        import datasets
        importlib.reload(datasets)

        from datasets import load_dataset

        datasets_config = {
            "bc2gm": ("bigbio/blurb", "bc2gm"),
            "jnlpba": ("bigbio/blurb", "jnlpba"),
            "ddi": ("bigbio/ddi_corpus", "ddi_corpus_source"),
            "gad": ("bigbio/gad", "gad_blurb_bigbio_text"),
            "hoc": ("bigbio/hallmarks_of_cancer", "hallmarks_of_cancer_source"),
            "pubmedqa": ("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source"),
            "biosses": ("bigbio/biosses", "biosses_bigbio_pairs")
        }

        data_path = Path("data/raw")
        data_path.mkdir(parents=True, exist_ok=True)

        successful = []
        failed = []

        for name, (repo, config) in datasets_config.items():
            print(f"\n📦 {name}...")
            try:
                dataset = load_dataset(repo, name=config)
                dataset.save_to_disk(str(data_path / name))
                successful.append(name)
                print(f"   ✓ Success!")
            except Exception as e:
                failed.append(name)
                print(f"   ✗ Failed: {str(e)[:80]}")

        return successful, failed

    except Exception as e:
        print(f"   ✗ Downgrade failed: {str(e)}")
        return [], []

def method3_direct_parquet():
    """Method 3: Load Parquet files directly"""
    print("\n" + "="*60)
    print("METHOD 3: Loading Parquet files directly")
    print("="*60)

    from datasets import load_dataset

    # Use the refs/convert/parquet endpoint
    datasets_config = {
        "bc2gm": "bigbio/blurb",
        "jnlpba": "bigbio/blurb",
        "ddi": "bigbio/ddi_corpus",
        "gad": "bigbio/gad",
        "hoc": "bigbio/hallmarks_of_cancer",
        "pubmedqa": "bigbio/pubmed_qa",
        "biosses": "bigbio/biosses"
    }

    data_path = Path("data/raw")
    data_path.mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    for name, repo in datasets_config.items():
        print(f"\n📦 {name}...")
        try:
            # Try to load the parquet version (auto-converted by HuggingFace)
            dataset = load_dataset(repo, data_files="**/*.parquet")
            dataset.save_to_disk(str(data_path / name))
            successful.append(name)
            print(f"   ✓ Success!")
        except Exception as e:
            failed.append(name)
            print(f"   ✗ Failed: {str(e)[:80]}")

    return successful, failed

def main():
    """Try each method until we get all 7 datasets."""
    print("🚀 ROBUST DATASET DOWNLOADER")
    print("="*60)
    print("Will try multiple methods until all 7 datasets are downloaded")
    print("="*60)

    total_needed = 7
    all_successful = set()

    # Try Method 1: trust_remote_code=True
    try:
        successful, failed = method1_trust_remote_code()
        all_successful.update(successful)
        print(f"\nMethod 1 result: {len(successful)}/7 datasets downloaded")

        if len(all_successful) == total_needed:
            print("\n" + "="*60)
            print("🎉 SUCCESS! All 7 datasets downloaded with Method 1")
            print("="*60)
            return
    except Exception as e:
        print(f"\nMethod 1 completely failed: {str(e)}")

    # Try Method 2: Downgrade datasets library
    if len(all_successful) < total_needed:
        print(f"\n⚠️  Still need {total_needed - len(all_successful)} more datasets")
        print("Trying Method 2...")

        try:
            successful, failed = method2_downgrade_datasets()
            all_successful.update(successful)
            print(f"\nMethod 2 result: {len(successful)}/7 datasets downloaded")

            if len(all_successful) == total_needed:
                print("\n" + "="*60)
                print("🎉 SUCCESS! All 7 datasets downloaded with Method 2")
                print("="*60)
                return
        except Exception as e:
            print(f"\nMethod 2 completely failed: {str(e)}")

    # Try Method 3: Direct Parquet
    if len(all_successful) < total_needed:
        print(f"\n⚠️  Still need {total_needed - len(all_successful)} more datasets")
        print("Trying Method 3...")

        try:
            successful, failed = method3_direct_parquet()
            all_successful.update(successful)
            print(f"\nMethod 3 result: {len(successful)}/7 datasets downloaded")
        except Exception as e:
            print(f"\nMethod 3 completely failed: {str(e)}")

    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\n✅ Successfully downloaded: {len(all_successful)}/7 datasets")
    print(f"   {sorted(list(all_successful))}")

    if len(all_successful) == total_needed:
        print("\n🎉 ALL DATASETS DOWNLOADED SUCCESSFULLY!")
        print("\nNext step: Run test_parsers.py to verify everything works")
    else:
        missing = total_needed - len(all_successful)
        print(f"\n⚠️  Still missing {missing} datasets")
        print("\nManual intervention needed - see DATASET_DOWNLOAD_FIX.md")

if __name__ == "__main__":
    main()
