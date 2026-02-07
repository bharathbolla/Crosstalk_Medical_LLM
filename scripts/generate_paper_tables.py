"""Auto-generate LaTeX tables for paper from results JSON.

Generates all tables:
- Table 1: Main results (5 tasks × 6 models)
- Table 2: Transfer matrix (5×5 heatmap)
- Table 3: Architecture ablation (A1-A4)
- Table 4: Token-controlled baseline (RQ5)
- Table 5: Efficiency comparison

Usage:
    python generate_paper_tables.py --results-dir results/ --output-dir paper/tables/
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.results.manager import ResultManager
from src.results.latex_generator import LatexTableGenerator


def generate_main_results_table(
    result_manager: ResultManager,
    output_path: Path,
):
    """Generate Table 1: Main results."""

    print("Generating Table 1: Main Results...")

    # Load all results
    all_results = result_manager.load_all_results(
        strategy_filter=["S1", "S2", "S3a", "S3b"],
    )

    # Build table
    table = result_manager.build_main_results_table(all_results)

    # Generate LaTeX
    generator = LatexTableGenerator()
    latex = generator.generate_main_results_table(table)

    # Save
    output_file = output_path / "table1_main_results.tex"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write(latex)

    print(f"  Saved to: {output_file}")


def generate_transfer_matrix_table(
    result_manager: ResultManager,
    output_path: Path,
):
    """Generate Table 2: Transfer matrix."""

    print("Generating Table 2: Transfer Matrix...")

    # Load single-task and multi-task results
    st_results = result_manager.load_all_results(strategy_filter=["S1"])
    mt_results = result_manager.load_all_results(strategy_filter=["S3b"])

    # Build transfer matrix
    transfer_matrix = result_manager.build_transfer_matrix(st_results, mt_results)

    # Generate LaTeX
    generator = LatexTableGenerator()
    latex = generator.generate_transfer_matrix_table(transfer_matrix)

    # Save
    output_file = output_path / "table2_transfer_matrix.tex"
    with open(output_file, "w") as f:
        f.write(latex)

    print(f"  Saved to: {output_file}")


def generate_ablation_table(
    result_manager: ResultManager,
    output_path: Path,
):
    """Generate Table 3: Architecture ablation."""

    print("Generating Table 3: Ablation Study...")

    # Load ablation results
    ablation_results = result_manager.load_all_results(
        strategy_filter=["A1", "A2", "A3", "A4"],
    )

    # Generate LaTeX
    generator = LatexTableGenerator()
    latex = generator.generate_ablation_table(ablation_results)

    # Save
    output_file = output_path / "table3_ablation.tex"
    with open(output_file, "w") as f:
        f.write(latex)

    print(f"  Saved to: {output_file}")


def generate_token_controlled_table(
    result_manager: ResultManager,
    output_path: Path,
):
    """Generate Table 4: Token-controlled baseline (RQ5)."""

    print("Generating Table 4: Token-Controlled Baseline...")

    # Load results
    st_results = result_manager.load_all_results(strategy_filter=["S1"])
    mt_results = result_manager.load_all_results(strategy_filter=["S3b"])
    tc_results = result_manager.load_all_results(strategy_filter=["S1_token_controlled"])

    # Build comparison table
    comparison = result_manager.build_token_comparison(st_results, mt_results, tc_results)

    # Generate LaTeX
    generator = LatexTableGenerator()
    latex = generator.generate_token_controlled_table(comparison)

    # Save
    output_file = output_path / "table4_token_controlled.tex"
    with open(output_file, "w") as f:
        f.write(latex)

    print(f"  Saved to: {output_file}")


def generate_efficiency_table(
    efficiency_results: Dict,
    output_path: Path,
):
    """Generate Table 5: Efficiency comparison."""

    print("Generating Table 5: Efficiency Comparison...")

    # Load inference benchmark results
    benchmark_file = Path("results/inference_bench/benchmark_results.json")

    if not benchmark_file.exists():
        print("  WARNING: Benchmark results not found. Skipping.")
        return

    with open(benchmark_file) as f:
        results = json.load(f)

    # Generate LaTeX
    generator = LatexTableGenerator()
    latex = generator.generate_efficiency_table(results)

    # Save
    output_file = output_path / "table5_efficiency.tex"
    with open(output_file, "w") as f:
        f.write(latex)

    print(f"  Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument("--results-dir", type=str, default="results/",
                        help="Results directory")
    parser.add_argument("--output-dir", type=str, default="paper/tables/",
                        help="Output directory for LaTeX tables")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Initialize result manager
    result_manager = ResultManager(results_dir=results_dir)

    # Generate all tables
    print("=" * 80)
    print("GENERATING PAPER TABLES")
    print("=" * 80)

    try:
        generate_main_results_table(result_manager, output_dir)
    except Exception as e:
        print(f"  Error: {e}")

    try:
        generate_transfer_matrix_table(result_manager, output_dir)
    except Exception as e:
        print(f"  Error: {e}")

    try:
        generate_ablation_table(result_manager, output_dir)
    except Exception as e:
        print(f"  Error: {e}")

    try:
        generate_token_controlled_table(result_manager, output_dir)
    except Exception as e:
        print(f"  Error: {e}")

    try:
        generate_efficiency_table({}, output_dir)
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 80)
    print("TABLE GENERATION COMPLETE")
    print("=" * 80)
    print(f"Tables saved to: {output_dir}")


if __name__ == "__main__":
    main()
