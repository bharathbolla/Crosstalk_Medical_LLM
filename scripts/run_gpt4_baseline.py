"""GPT-4o-mini API baseline (reference point, not a true baseline).

IMPORTANT: Frame this honestly in the paper as a reference point,
not a fair comparison. API models have advantages:
- Massive scale (100B+ params)
- Continuous updates
- Different training objectives

Usage:
    python run_gpt4_baseline.py --task semeval2017t3 --n-samples 100

    # All tasks (expensive!)
    python run_gpt4_baseline.py --tasks all --n-samples 500
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List
import json
import os
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.base import TaskRegistry
from src.evaluation.metrics import compute_task_metrics


def call_gpt4_api(prompt: str, max_tokens: int = 100) -> str:
    """Call GPT-4o-mini API.

    TODO: Implement with anthropic SDK or openai SDK
    """
    # Placeholder implementation
    # In reality, use:
    # from openai import OpenAI
    # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # response = client.chat.completions.create(...)

    raise NotImplementedError("Implement GPT-4 API call")


def format_prompt_for_task(task_name: str, sample: dict) -> str:
    """Format task-specific prompt for GPT-4."""

    # TODO: Task-specific prompts
    if task_name == "semeval2017t3":
        return f"""You are a medical question answering system.

Question: {sample.input_text}

Answer: [Your answer here]
"""
    elif task_name == "semeval2014t7":
        return f"""You are a medical named entity recognition system.
Extract disorder mentions from the following clinical text.

Text: {sample.input_text}

Disorders: [List disorder mentions]
"""
    else:
        raise NotImplementedError(f"Prompt not implemented for {task_name}")


def evaluate_gpt4(
    task_name: str,
    n_samples: int,
    output_dir: Path,
) -> Dict[str, float]:
    """Evaluate GPT-4o-mini on task."""

    print(f"\nEvaluating GPT-4o-mini on {task_name}")
    print("=" * 80)

    # Load test data
    print("Loading data...")
    task_registry = TaskRegistry()
    test_dataset = task_registry.get_dataset(task_name, split="test")

    # Sample if needed
    if n_samples and n_samples < len(test_dataset):
        import random
        test_dataset = random.sample(test_dataset, n_samples)

    print(f"Evaluating on {len(test_dataset)} samples")

    # Get predictions
    print("Querying GPT-4 API...")
    predictions = []
    labels = []

    for sample in tqdm(test_dataset):
        prompt = format_prompt_for_task(task_name, sample)

        try:
            response = call_gpt4_api(prompt)
            # TODO: Parse response into structured prediction
            predictions.append(response)
            labels.append(sample.labels)
        except Exception as e:
            print(f"API error: {e}")
            continue

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_task_metrics(
        task_name=task_name,
        predictions=predictions,
        labels=labels,
    )

    print(f"\nResults:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    output_file = output_dir / f"gpt4_{task_name}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({
            "model": "gpt-4o-mini",
            "task": task_name,
            "n_samples": len(predictions),
            "metrics": metrics,
        }, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="GPT-4o-mini baseline")
    parser.add_argument("--task", type=str, help="Specific task")
    parser.add_argument("--tasks", type=str, default=None, help="all or specific tasks")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples (None = all)")
    parser.add_argument("--output-dir", type=str, default="results/gpt4",
                        help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Check API key
    if "OPENAI_API_KEY" not in os.environ:
        print("ERROR: OPENAI_API_KEY not set")
        print("Set with: export OPENAI_API_KEY=sk-...")
        return 1

    # Determine tasks
    if args.tasks == "all":
        tasks = ["semeval2014t7", "semeval2015t14", "semeval2016t12",
                 "semeval2017t3", "semeval2021t6"]
    elif args.task:
        tasks = [args.task]
    else:
        tasks = ["semeval2017t3"]  # default to easiest task

    # Run evaluation
    print("=" * 80)
    print("GPT-4o-mini BASELINE (Reference Point)")
    print("=" * 80)
    print(f"Tasks: {len(tasks)}")
    print(f"Samples per task: {args.n_samples or 'all'}")
    print("\nNOTE: This is a reference point, not a fair comparison.")
    print("      GPT-4 has >100B params, continuous updates, etc.")
    print("=" * 80)

    results = {}
    for task_name in tasks:
        try:
            metrics = evaluate_gpt4(task_name, args.n_samples, output_dir)
            results[task_name] = metrics
        except Exception as e:
            print(f"Error evaluating {task_name}: {e}")
            continue

    # Summary
    print("\n" + "=" * 80)
    print("GPT-4 SUMMARY")
    print("=" * 80)

    for task_name, metrics in results.items():
        primary_metric = list(metrics.keys())[0]
        print(f"{task_name}: {primary_metric}={metrics[primary_metric]:.4f}")


if __name__ == "__main__":
    main()
