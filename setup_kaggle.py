"""
Kaggle Environment Setup Script
Creates isolated virtual environment to avoid dependency conflicts
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print(f"Error: {result.stderr}")
        return False

    if result.stdout:
        print(result.stdout)

    print(f"✅ Success: {description}")
    return True

def main():
    print("\n" + "="*60)
    print("KAGGLE ENVIRONMENT SETUP")
    print("="*60)
    print("\nThis will create an isolated Python environment")
    print("to avoid dependency conflicts with Kaggle's packages.\n")

    # Step 1: Create venv
    if not run_command(
        "python3 -m venv venv",
        "Step 1: Creating virtual environment"
    ):
        sys.exit(1)

    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        python_cmd = "venv\\Scripts\\python"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Linux/Mac
        activate_cmd = "source venv/bin/activate"
        python_cmd = "venv/bin/python"
        pip_cmd = "venv/bin/pip"

    # Step 2: Upgrade pip
    if not run_command(
        f"{pip_cmd} install --upgrade pip -q",
        "Step 2: Upgrading pip"
    ):
        sys.exit(1)

    # Step 3: Install compatible packages
    print("\n" + "="*60)
    print("Step 3: Installing compatible packages")
    print("="*60)

    packages = [
        ("pyarrow==14.0.0", "Apache Arrow library"),
        ("datasets==2.20.0", "HuggingFace Datasets"),
        ("transformers==4.40.0", "HuggingFace Transformers"),
        ("evaluate==0.4.2", "Evaluation library"),
        ("torch", "PyTorch (latest)"),
        ("accelerate==0.30.0", "Training acceleration"),
        ("scikit-learn==1.5.0", "ML utilities"),
        ("pyyaml==6.0.1", "YAML parser"),
    ]

    for package, description in packages:
        print(f"\nInstalling {description}...")
        result = subprocess.run(
            f"{pip_cmd} install -q {package}",
            shell=True,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"  ⚠️  Warning: Failed to install {package}")
            print(f"  Error: {result.stderr[:200]}")
        else:
            print(f"  ✓ Installed {package}")

    # Step 4: Verify installation
    print("\n" + "="*60)
    print("Step 4: Verifying installation")
    print("="*60)

    verify_script = """
import sys
print(f"Python: {sys.version}")
import datasets
print(f"datasets: {datasets.__version__}")
import pyarrow
print(f"pyarrow: {pyarrow.__version__}")
import transformers
print(f"transformers: {transformers.__version__}")
"""

    with open("_verify.py", "w") as f:
        f.write(verify_script)

    result = subprocess.run(
        f"{python_cmd} _verify.py",
        shell=True,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    os.remove("_verify.py")

    # Final instructions
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\nTo use this environment on Kaggle:")
    print("")
    print("1. In each notebook cell, activate the environment:")
    print(f"   !{activate_cmd}")
    print("")
    print("2. Then run your commands:")
    print(f"   !{python_cmd} test_parsers.py")
    print("")
    print("Or in Python cells:")
    print("   import sys")
    print(f"   sys.path.insert(0, 'venv/lib/python3.12/site-packages')")
    print("   sys.path.insert(0, 'src')")
    print("")
    print("="*60)

if __name__ == "__main__":
    main()
