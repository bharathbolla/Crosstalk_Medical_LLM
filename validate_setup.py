"""Complete setup validation script.

Checks:
1. Python version
2. Dependencies installed
3. Environment variables set
4. HuggingFace authentication
5. Directory structure
6. Ready to download datasets

Usage:
    python validate_setup.py
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.10."""
    print("\n[1/7] Checking Python version...")

    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  [X] Python {version.major}.{version.minor} (need 3.10+)")
        return False


def check_dependencies():
    """Check if key dependencies are installed."""
    print("\n[2/7] Checking dependencies...")

    required_packages = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "wandb",
        "hydra",
        "numpy",
        "pandas",
        "dotenv",
    ]

    missing = []
    installed = []

    for package in required_packages:
        try:
            if package == "dotenv":
                __import__("dotenv")
            elif package == "hydra":
                __import__("hydra")
            else:
                __import__(package)
            installed.append(package)
        except ImportError:
            missing.append(package)

    if not missing:
        print(f"  [OK] All {len(installed)} required packages installed")
        return True
    else:
        print(f"  [X] Missing {len(missing)} packages: {', '.join(missing)}")
        print(f"\n  Install with:")
        print(f"    pip install -e .")
        return False


def check_env_file():
    """Check if .env file exists."""
    print("\n[3/7] Checking .env file...")

    env_path = Path(".env")

    if not env_path.exists():
        print("  [X] .env file not found")
        print("  Create with: cp .env.template .env")
        return False

    print("  [OK] .env file exists")
    return True


def check_env_variables():
    """Check if environment variables are set."""
    print("\n[4/7] Checking environment variables...")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("  [X] python-dotenv not installed")
        return False

    required_vars = ["HF_TOKEN", "WANDB_API_KEY"]
    optional_vars = ["WANDB_ENTITY", "WANDB_PROJECT"]

    all_ok = True

    for var in required_vars:
        value = os.getenv(var, "")
        is_placeholder = ("xxx" in value.lower() or value.count("x") > 10 or
                         "your-" in value.lower() or value == "")

        if not value:
            print(f"  [X] {var}: NOT SET")
            all_ok = False
        elif is_placeholder:
            print(f"  [!] {var}: PLACEHOLDER (needs real value)")
            all_ok = False
        else:
            print(f"  [OK] {var}: SET (length: {len(value)})")

    for var in optional_vars:
        value = os.getenv(var, "")
        if value and "xxx" not in value and "your-" not in value:
            print(f"  [OK] {var}: SET")

    if not all_ok:
        print("\n  Fix by editing .env file with your actual tokens")
        print("  Get HF_TOKEN from: https://huggingface.co/settings/tokens")
        print("  Get WANDB_API_KEY from: https://wandb.ai/authorize")

    return all_ok


def check_hf_auth():
    """Check HuggingFace authentication."""
    print("\n[5/7] Checking HuggingFace authentication...")

    try:
        from dotenv import load_dotenv
        from huggingface_hub import HfApi

        load_dotenv()
        token = os.getenv("HF_TOKEN")

        if not token or "xxx" in token:
            print("  [X] HF_TOKEN not set properly")
            return False

        api = HfApi(token=token)
        user = api.whoami()

        print(f"  [OK] Authenticated as: {user['name']}")
        return True

    except ImportError:
        print("  [!] huggingface_hub not installed (will install with dependencies)")
        return None  # Not critical yet
    except Exception as e:
        print(f"  [X] Authentication failed: {str(e)[:60]}")
        print("  Check your HF_TOKEN in .env file")
        return False


def check_directory_structure():
    """Check if required directories exist."""
    print("\n[6/7] Checking directory structure...")

    required_dirs = [
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        "src/results",
        "configs/model",
        "configs/task",
        "configs/strategy",
        "scripts",
        "tests",
    ]

    all_exist = True

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            file_count = len(list(path.glob("*.py"))) + len(list(path.glob("*.yaml")))
            if file_count > 0:
                print(f"  [OK] {dir_path}/ ({file_count} files)")
        else:
            print(f"  [X] {dir_path}/ NOT FOUND")
            all_exist = False

    return all_exist


def check_download_ready():
    """Check if ready to download datasets."""
    print("\n[7/7] Checking dataset download readiness...")

    try:
        from datasets import load_dataset

        print("  [OK] 'datasets' library installed")
        print("  [OK] Ready to download datasets!")
        return True

    except ImportError:
        print("  [!] 'datasets' library not installed yet")
        print("  Will be installed with: pip install -e .")
        return None  # Not installed yet


def main():
    """Run all checks."""
    print("=" * 60)
    print("SETUP VALIDATION")
    print("=" * 60)

    results = {
        "Python version": check_python_version(),
        "Dependencies": check_dependencies(),
        ".env file": check_env_file(),
        "Environment variables": check_env_variables(),
        "HF authentication": check_hf_auth(),
        "Directory structure": check_directory_structure(),
        "Download ready": check_download_ready(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60 + "\n")

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for check, result in results.items():
        if result is True:
            print(f"  [OK] {check}")
        elif result is False:
            print(f"  [X] {check}")
        else:
            print(f"  [!] {check} (will be fixed)")

    print(f"\nResults: {passed} passed, {failed} failed, {skipped} pending")

    # Next steps
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60 + "\n")

    if failed == 0:
        print("Ready to go! Next:")
        print("  1. Download datasets:")
        print("     python scripts/download_datasets_hf.py --all")
        print("  2. Start experiments!")

    else:
        print("Fix the following:")

        if not results["Dependencies"]:
            print("  1. Install dependencies:")
            print("     pip install -e .")

        if not results[".env file"] or not results["Environment variables"]:
            print("  2. Set up .env file:")
            print("     - Copy: cp .env.template .env")
            print("     - Edit .env with your tokens")
            print("     - HF_TOKEN: https://huggingface.co/settings/tokens")
            print("     - WANDB_API_KEY: https://wandb.ai/authorize")

        if not results["HF authentication"]:
            print("  3. Fix HuggingFace token in .env")

        print("\n  Then run this script again: python validate_setup.py")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
