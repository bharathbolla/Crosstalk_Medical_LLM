"""Run linear probing tasks on trained adapters.

Tests what medical knowledge was learned in the shared adapter.

Four probes:
1. Medical concept type (5-class)
2. Negation detection (binary)
3. Abbreviation expansion (accuracy)
4. Temporal ordering (3-class)

Usage:
    # Single adapter
    python run_probing.py --checkpoint checkpoints/s3b_llama3b/best

    # All adapters
    python run_probing.py --all
"""

import sys
import argparse
from pathlib import Path
from typing import Dict
import json

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_loader import load_model
from src.models.adapters import SharedPrivateLoRA
from src.evaluation.probing import LinearProbe, evaluate_adapter_probes


def run_probing(
    checkpoint_path: Path,
    probe_datasets: Dict[str, any],
    device: torch.device,
) -> Dict[str, float]:
    """Run all probing tasks on adapter."""

    print(f"\nProbing adapter: {checkpoint_path}")
    print("=" * 80)

    # Load model with adapter
    print("Loading model...")
    # TODO: Load model from checkpoint

    # Run probes
    print("Running probing tasks...")
    results = evaluate_adapter_probes(
        model=None,  # TODO
        adapter_name="shared",
        probe_datasets=probe_datasets,
        device=device,
    )

    print("\nProbing Results:")
    for probe_name, accuracy in results.items():
        print(f"  {probe_name}: {accuracy:.4f}")

    return results


def load_probe_datasets() -> Dict[str, any]:
    """Load all 4 probe datasets.

    TODO: Implement actual loading from:
    - UMLS concept types
    - NegEx negation examples
    - CASI abbreviation pairs
    - TimeML temporal relations
    """
    print("Loading probe datasets...")

    # Placeholder
    probe_datasets = {
        "medical_concept_type": None,
        "negation_detection": None,
        "abbreviation_expansion": None,
        "temporal_ordering": None,
    }

    return probe_datasets


def main():
    parser = argparse.ArgumentParser(description="Adapter probing")
    parser.add_argument("--all", action="store_true", help="Probe all checkpoints")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/probing",
                        help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load probe datasets
    probe_datasets = load_probe_datasets()

    # Determine checkpoints
    if args.all:
        checkpoint_dir = Path("checkpoints")
        checkpoints = list(checkpoint_dir.glob("*/best"))
    else:
        checkpoints = [Path(args.checkpoint)]

    # Run probing
    print("=" * 80)
    print("ADAPTER PROBING")
    print("=" * 80)
    print(f"Checkpoints: {len(checkpoints)}")
    print("=" * 80)

    results = {}

    for checkpoint_path in checkpoints:
        try:
            probe_results = run_probing(
                checkpoint_path=checkpoint_path,
                probe_datasets=probe_datasets,
                device=device,
            )
            results[str(checkpoint_path)] = probe_results
        except Exception as e:
            print(f"Error probing {checkpoint_path}: {e}")
            continue

    # Save results
    output_file = output_dir / "probing_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("PROBING SUMMARY")
    print("=" * 80)

    for checkpoint, probe_results in results.items():
        print(f"\n{checkpoint}:")
        for probe_name, accuracy in probe_results.items():
            print(f"  {probe_name}: {accuracy:.4f}")


if __name__ == "__main__":
    main()
