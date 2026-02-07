"""Phase 0: Contamination detection with 3-layer protocol.

Checks if test data was in pretraining corpus for all models and tasks.

Usage:
    # All models and tasks
    python run_contamination_check.py --all

    # Specific model
    python run_contamination_check.py --model llama3b --tasks all

    # Specific task
    python run_contamination_check.py --model llama3b --task semeval2017t3
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.base import TaskRegistry
from src.models.base_loader import load_model
from src.evaluation.contamination import ContaminationChecker, ContaminationResult


MODELS = [
    "google/gemma-2-2b",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Llama-3.2-3B",
    "mistralai/Mistral-7B-v0.3",
    "meta-llama/Llama-3.1-8B",
    "BioMistral/BioMistral-7B",
]

TASKS = [
    "semeval2014t7",
    "semeval2015t14",
    "semeval2016t12",
    "semeval2017t3",
    "semeval2021t6",
]


def check_contamination(
    model_name: str,
    task_name: str,
    output_dir: Path,
) -> ContaminationResult:
    """Run 3-layer contamination check."""

    print(f"\nChecking contamination: {model_name} on {task_name}")
    print("=" * 80)

    # Load model (no fine-tuning)
    print("Loading base model...")
    model = load_model(model_name, quantization_config=None, lora_config=None)

    # Load test data
    print(f"Loading test data for {task_name}...")
    task_registry = TaskRegistry()
    test_data = task_registry.get_dataset(task_name, split="test")
    control_data = task_registry.get_dataset(task_name, split="control")  # random clinical text

    # Run contamination check
    print("\nRunning 3-layer contamination protocol...")
    checker = ContaminationChecker(model=model, tokenizer=model.tokenizer)

    result = checker.check(
        task_name=task_name,
        test_data=test_data,
        control_data=control_data,
    )

    # Print results
    print("\nContamination Report:")
    print(f"  Layer 1 (Zero-shot): {'⚠ FLAGGED' if result.zero_shot_flagged else '✓ CLEAN'}")
    print(f"    Base model F1: {result.zero_shot_score:.3f}")
    print(f"    SOTA: {result.sota_score:.3f}")
    print(f"    Threshold: {result.zero_shot_threshold:.3f}")

    print(f"\n  Layer 2 (N-gram): {'⚠ FLAGGED' if result.ngram_flagged else '✓ CLEAN'}")
    print(f"    Overlap ratio: {result.ngram_overlap_ratio:.3f}")
    print(f"    Threshold: {result.ngram_threshold:.3f}")

    print(f"\n  Layer 3 (Min-K%): {'⚠ FLAGGED' if result.mink_flagged else '✓ CLEAN'}")
    print(f"    P-value: {result.mink_pvalue:.4f}")
    print(f"    Threshold: {result.mink_threshold:.4f}")

    print(f"\n  OVERALL: {'⚠ CONTAMINATED' if result.is_contaminated else '✓ CLEAN'}")

    # Save result
    output_file = output_dir / f"{model_name.replace('/', '_')}_{task_name}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    print(f"\nSaved to: {output_file}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Contamination detection")
    parser.add_argument("--all", action="store_true", help="Check all models and tasks")
    parser.add_argument("--model", type=str, help="Specific model")
    parser.add_argument("--task", type=str, help="Specific task")
    parser.add_argument("--tasks", type=str, default="all", help="all or specific tasks")
    parser.add_argument("--output-dir", type=str, default="results/contamination",
                        help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Determine models and tasks to check
    if args.all:
        models = MODELS
        tasks = TASKS
    else:
        models = [args.model] if args.model else MODELS
        if args.task:
            tasks = [args.task]
        elif args.tasks == "all":
            tasks = TASKS
        else:
            tasks = args.tasks.split(",")

    # Run checks
    print("=" * 80)
    print("CONTAMINATION DETECTION")
    print("=" * 80)
    print(f"Models: {len(models)}")
    print(f"Tasks: {len(tasks)}")
    print(f"Total checks: {len(models) * len(tasks)}")
    print("=" * 80)

    results: Dict[str, Dict[str, ContaminationResult]] = {}

    for model_name in models:
        results[model_name] = {}
        for task_name in tasks:
            try:
                result = check_contamination(model_name, task_name, output_dir)
                results[model_name][task_name] = result
            except Exception as e:
                print(f"Error checking {model_name} on {task_name}: {e}")
                continue

    # Summary
    print("\n" + "=" * 80)
    print("CONTAMINATION SUMMARY")
    print("=" * 80)

    contaminated_count = 0
    total_count = 0

    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for task_name, result in model_results.items():
            total_count += 1
            if result.is_contaminated:
                contaminated_count += 1
                print(f"  ⚠ {task_name}: CONTAMINATED")
            else:
                print(f"  ✓ {task_name}: CLEAN")

    print(f"\nOverall: {contaminated_count}/{total_count} contaminated")

    if contaminated_count > 0:
        print("\n⚠ WARNING: Contamination detected. Consider:")
        print("  1. Excluding contaminated model-task pairs from experiments")
        print("  2. Replacing contaminated tasks with alternative datasets")
        print("  3. Reporting contamination in paper limitations")

    # Save summary
    summary_file = output_dir / "contamination_summary.json"
    with open(summary_file, "w") as f:
        summary = {
            "total_checks": total_count,
            "contaminated": contaminated_count,
            "clean": total_count - contaminated_count,
            "results": {
                model: {task: result.to_dict() for task, result in tasks.items()}
                for model, tasks in results.items()
            },
        }
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
