"""Run BERT-based baselines (BERT, BioBERT, ClinicalBERT).

Usage:
    # Single task
    python run_baseline.py --model bert-base-uncased --task semeval2017t3

    # All tasks
    python run_baseline.py --model dmis-lab/biobert-base-cased-v1.1 --tasks all

    # All baselines on all tasks
    python run_baseline.py --all
"""

import sys
import argparse
from pathlib import Path
from typing import Dict
import json

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.base import TaskRegistry
from src.evaluation.metrics import compute_task_metrics
from src.results.manager import ResultManager


BASELINE_MODELS = {
    "bert": "bert-base-uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
}

TASKS = [
    "semeval2014t7",
    "semeval2015t14",
    "semeval2016t12",
    "semeval2017t3",
    "semeval2021t6",
]


def train_baseline(
    model_name: str,
    task_name: str,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    """Train BERT-based baseline on single task."""

    print(f"\nTraining {model_name} on {task_name}")
    print("=" * 80)

    # Load tokenizer and model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=10,  # TODO: get from task config
    )
    model.to(device)

    # Load data
    print("Loading data...")
    task_registry = TaskRegistry()
    train_dataset = task_registry.get_dataset(task_name, split="train")
    dev_dataset = task_registry.get_dataset(task_name, split="dev")
    test_dataset = task_registry.get_dataset(task_name, split="test")

    # Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir / model_name.replace("/", "_") / task_name),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
    )

    # Train
    print("Training...")
    trainer.train()

    # Evaluate
    print("Evaluating on test set...")
    predictions = trainer.predict(test_dataset)
    labels = [sample.labels for sample in test_dataset]

    metrics = compute_task_metrics(
        task_name=task_name,
        predictions=predictions.predictions,
        labels=labels,
    )

    print(f"\nResults:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="BERT baselines")
    parser.add_argument("--all", action="store_true", help="Run all baselines on all tasks")
    parser.add_argument("--model", type=str, help="Specific model")
    parser.add_argument("--task", type=str, help="Specific task")
    parser.add_argument("--tasks", type=str, default="all", help="all or specific tasks")
    parser.add_argument("--output-dir", type=str, default="results/baselines",
                        help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine models and tasks
    if args.all:
        models = list(BASELINE_MODELS.values())
        tasks = TASKS
    else:
        models = [BASELINE_MODELS.get(args.model, args.model)] if args.model else list(BASELINE_MODELS.values())
        if args.task:
            tasks = [args.task]
        elif args.tasks == "all":
            tasks = TASKS
        else:
            tasks = args.tasks.split(",")

    # Run baselines
    print("=" * 80)
    print("BERT BASELINES")
    print("=" * 80)
    print(f"Models: {len(models)}")
    print(f"Tasks: {len(tasks)}")
    print("=" * 80)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model_name in models:
        results[model_name] = {}
        for task_name in tasks:
            try:
                metrics = train_baseline(model_name, task_name, output_dir, device)
                results[model_name][task_name] = metrics
            except Exception as e:
                print(f"Error training {model_name} on {task_name}: {e}")
                continue

    # Save results
    result_manager = ResultManager(results_dir=output_dir)
    for model_name, model_results in results.items():
        result_manager.save_result(
            experiment_id=f"baseline_{model_name.replace('/', '_')}",
            model_name=model_name,
            strategy="baseline",
            task_results=model_results,
        )

    print("\n" + "=" * 80)
    print("BASELINE SUMMARY")
    print("=" * 80)

    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for task_name, metrics in model_results.items():
            primary_metric = list(metrics.keys())[0]
            print(f"  {task_name}: {primary_metric}={metrics[primary_metric]:.4f}")


if __name__ == "__main__":
    main()
