"""Setup script for Medical Cross-Task Knowledge Transfer project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="medical-cross-task-transfer",
    version="0.1.0",
    author="Research Team",
    description="Cross-Task Knowledge Transfer with Small Language Models for Medical NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/medical-cross-task-transfer",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "bitsandbytes>=0.43.0",
        "datasets>=2.18.0",
        "accelerate>=0.28.0",
        "wandb>=0.16.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "seqeval>=1.2.2",
        "scikit-learn>=1.4.0",
        "scipy>=1.12.0",
        "numpy>=1.26.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.1.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
        ],
        "fast": [
            "flash-attn>=2.5.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="medical-nlp multi-task-learning knowledge-transfer small-language-models",
)
