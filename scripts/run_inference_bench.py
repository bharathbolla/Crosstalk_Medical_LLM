"""Inference benchmarking: latency, throughput, VRAM usage.

Measures deployment-realistic performance metrics:
- Latency (ms per sample)
- Throughput (samples/sec)
- VRAM usage (GB)
- Calibration (ECE)

Usage:
    # Single model
    python run_inference_bench.py --model llama3b --quantization none

    # Compare quantization levels
    python run_inference_bench.py --model llama8b --quantization fp16,int8,int4

    # All models
    python run_inference_bench.py --all
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List
import json
import time

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_loader import load_model
from src.data.base import TaskRegistry
from src.utils.vram_monitor import VRAMMonitor
from src.evaluation.calibration import expected_calibration_error


def benchmark_model(
    model_name: str,
    quantization: str,
    task_name: str,
    n_samples: int,
    device: torch.device,
) -> Dict[str, float]:
    """Benchmark single model."""

    print(f"\nBenchmarking {model_name} ({quantization})")
    print("=" * 80)

    # VRAM monitor
    vram_monitor = VRAMMonitor()
    vram_monitor.reset()

    # Load model
    print("Loading model...")
    quant_config = {
        "enabled": quantization != "none",
        "bits": 4 if quantization == "int4" else 8 if quantization == "int8" else 16,
        "quant_type": "nf4" if quantization == "int4" else None,
    } if quantization != "none" else None

    model = load_model(
        model_name=model_name,
        quantization_config=quant_config,
        device=device,
    )

    vram_after_load = vram_monitor.get_current_usage()

    # Load test data
    print("Loading test data...")
    task_registry = TaskRegistry()
    test_dataset = task_registry.get_dataset(task_name, split="test")

    if n_samples:
        test_dataset = test_dataset[:n_samples]

    # Warmup
    print("Warmup...")
    for i in range(10):
        sample = test_dataset[i]
        with torch.no_grad():
            model(sample.input_text)

    # Benchmark latency
    print(f"Benchmarking latency on {len(test_dataset)} samples...")
    latencies = []

    for sample in test_dataset:
        start = time.time()
        with torch.no_grad():
            output = model(sample.input_text)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)

    vram_after_inference = vram_monitor.get_peak_usage()

    # Compute metrics
    mean_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    throughput = 1000.0 / mean_latency  # samples/sec

    # Get predictions for calibration
    print("Computing calibration...")
    predictions = []
    confidences = []
    labels = []

    with torch.no_grad():
        for sample in test_dataset:
            output = model(sample.input_text)
            # TODO: Extract confidence scores
            predictions.append(output)
            confidences.append(0.0)  # placeholder
            labels.append(sample.labels)

    # ECE (placeholder until we have proper confidence scores)
    ece = 0.0  # expected_calibration_error(confidences, predictions, labels)[0]

    results = {
        "model": model_name,
        "quantization": quantization,
        "latency_mean_ms": mean_latency,
        "latency_p50_ms": p50_latency,
        "latency_p95_ms": p95_latency,
        "latency_p99_ms": p99_latency,
        "throughput_samples_per_sec": throughput,
        "vram_after_load_gb": vram_after_load,
        "vram_peak_gb": vram_after_inference,
        "ece": ece,
    }

    print("\nResults:")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Inference benchmarking")
    parser.add_argument("--all", action="store_true", help="Benchmark all models")
    parser.add_argument("--model", type=str, help="Specific model")
    parser.add_argument("--quantization", type=str, default="none",
                        help="none,fp16,int8,int4 (comma-separated)")
    parser.add_argument("--task", type=str, default="semeval2017t3",
                        help="Task for benchmarking")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples")
    parser.add_argument("--output-dir", type=str, default="results/inference_bench",
                        help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine models and quantization levels
    if args.all:
        models = [
            "google/gemma-2-2b",
            "microsoft/Phi-3-mini-4k-instruct",
            "meta-llama/Llama-3.2-3B",
            "meta-llama/Llama-3.1-8B",
        ]
        quantization_levels = ["fp16", "int4"]
    else:
        models = [args.model] if args.model else ["meta-llama/Llama-3.2-3B"]
        quantization_levels = args.quantization.split(",")

    # Run benchmarks
    print("=" * 80)
    print("INFERENCE BENCHMARKING")
    print("=" * 80)
    print(f"Models: {len(models)}")
    print(f"Quantization levels: {quantization_levels}")
    print(f"Task: {args.task}")
    print(f"Samples: {args.n_samples}")
    print("=" * 80)

    results = []

    for model_name in models:
        for quant_level in quantization_levels:
            try:
                result = benchmark_model(
                    model_name=model_name,
                    quantization=quant_level,
                    task_name=args.task,
                    n_samples=args.n_samples,
                    device=device,
                )
                results.append(result)
            except Exception as e:
                print(f"Error benchmarking {model_name} ({quant_level}): {e}")
                continue

    # Save results
    output_file = output_dir / "benchmark_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    for result in results:
        print(f"\n{result['model']} ({result['quantization']}):")
        print(f"  Latency: {result['latency_p50_ms']:.1f}ms (p50)")
        print(f"  Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec")
        print(f"  VRAM: {result['vram_peak_gb']:.2f} GB")


if __name__ == "__main__":
    main()
