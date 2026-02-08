"""
Analyze experiment results and generate tables/figures for paper.
Run this after downloading results from Kaggle.

Usage:
    python analyze_results.py --results_dir results/
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from collections import defaultdict


def load_all_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all result JSON files from directory."""
    all_results = []

    for result_file in sorted(results_dir.glob("results_*.json")):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {result_file}: {e}")

    return all_results


def create_comparison_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Create comparison DataFrame from results."""
    rows = []

    for r in results:
        config = r.get('config', {})
        test_results = r.get('test_results', {})
        train_results = r.get('train_results', {})

        # Handle both single and multiple datasets
        datasets = config.get('datasets', [])
        dataset_str = '+'.join(datasets) if len(datasets) > 1 else datasets[0] if datasets else 'unknown'

        row = {
            'experiment_id': r.get('experiment_id', 'unknown'),
            'experiment_type': config.get('experiment_type', 'unknown'),
            'model': config.get('model_name', 'unknown').split('/')[-1],  # Short name
            'dataset': dataset_str,
            'num_datasets': len(datasets),

            # Performance metrics
            'f1': test_results.get('eval_f1', 0.0),
            'precision': test_results.get('eval_precision', 0.0),
            'recall': test_results.get('eval_recall', 0.0),
            'loss': test_results.get('eval_loss', 0.0),

            # Training info
            'epochs': config.get('num_epochs', 0),
            'batch_size': config.get('batch_size', 0),
            'learning_rate': config.get('learning_rate', 0.0),

            # Efficiency metrics
            'tokens': r.get('token_count', 0),
            'train_time': train_results.get('train_runtime', 0.0),
            'samples_per_sec': train_results.get('train_samples_per_second', 0.0),

            # Model size
            'trainable_params': r.get('model_params', {}).get('trainable', 0),
            'total_params': r.get('model_params', {}).get('total', 0),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def generate_single_task_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table for single-task baselines."""
    st_df = df[df['experiment_type'].str.contains('single', case=False, na=False)]

    if len(st_df) == 0:
        return "No single-task results found."

    # Pivot: rows=datasets, cols=models, values=F1
    pivot = st_df.pivot_table(
        index='dataset',
        columns='model',
        values='f1',
        aggfunc='mean'  # Average if multiple runs
    )

    latex = pivot.to_latex(
        float_format="%.3f",
        caption="Single-Task Baseline Results (F1 Scores)",
        label="tab:single_task"
    )

    return latex


def generate_multitask_comparison(df: pd.DataFrame) -> str:
    """Generate comparison table: single-task vs multi-task."""

    # Get single-task and multi-task results
    st_df = df[df['experiment_type'].str.contains('single', case=False, na=False)]
    mt_df = df[df['experiment_type'].str.contains('multi', case=False, na=False)]

    if len(st_df) == 0 or len(mt_df) == 0:
        return "Need both single-task and multi-task results for comparison."

    comparison = []

    for dataset in st_df['dataset'].unique():
        st_f1 = st_df[st_df['dataset'] == dataset]['f1'].mean()
        mt_f1 = mt_df[mt_df['dataset'] == dataset]['f1'].mean()

        comparison.append({
            'Dataset': dataset,
            'Single-Task F1': st_f1,
            'Multi-Task F1': mt_f1,
            'Improvement': mt_f1 - st_f1,
            'Relative Gain (%)': 100 * (mt_f1 - st_f1) / st_f1 if st_f1 > 0 else 0,
        })

    comp_df = pd.DataFrame(comparison)

    latex = comp_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="Multi-Task vs Single-Task Comparison",
        label="tab:mt_vs_st"
    )

    return latex


def generate_token_controlled_analysis(df: pd.DataFrame) -> str:
    """
    Critical RQ5 analysis: Multi-task vs token-controlled baseline.
    This answers whether gains are from data exposure or genuine transfer.
    """
    mt_df = df[df['experiment_type'].str.contains('multi', case=False, na=False)]
    tc_df = df[df['experiment_type'].str.contains('token_controlled', case=False, na=False)]

    if len(mt_df) == 0 or len(tc_df) == 0:
        return "Need both multi-task and token-controlled results for RQ5 analysis."

    analysis = []

    for dataset in mt_df['dataset'].unique():
        mt_data = mt_df[mt_df['dataset'] == dataset].iloc[0]
        tc_data = tc_df[tc_df['dataset'] == dataset]

        if len(tc_data) == 0:
            continue

        tc_data = tc_data.iloc[0]

        # Key insight: If multi-task STILL better with equal tokens → genuine transfer!
        mt_f1 = mt_data['f1']
        tc_f1 = tc_data['f1']
        mt_tokens = mt_data['tokens']
        tc_tokens = tc_data['tokens']

        analysis.append({
            'Dataset': dataset,
            'Multi-Task F1': mt_f1,
            'Token-Controlled F1': tc_f1,
            'MT Tokens': mt_tokens,
            'TC Tokens': tc_tokens,
            'Token Parity': abs(mt_tokens - tc_tokens) / mt_tokens < 0.05,
            'F1 Difference': mt_f1 - tc_f1,
            'Genuine Transfer?': 'YES' if (mt_f1 > tc_f1 and abs(mt_tokens - tc_tokens) / mt_tokens < 0.05) else 'NO',
        })

    analysis_df = pd.DataFrame(analysis)

    latex = analysis_df.to_latex(
        index=False,
        float_format="%.3f",
        caption="RQ5: Token-Controlled Analysis (Evidence of Genuine Transfer)",
        label="tab:rq5_token_controlled"
    )

    return latex


def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistical tests for paper."""
    from scipy import stats

    st_df = df[df['experiment_type'].str.contains('single', case=False, na=False)]
    mt_df = df[df['experiment_type'].str.contains('multi', case=False, na=False)]

    stats_results = {}

    if len(st_df) > 0 and len(mt_df) > 0:
        # T-test
        t_stat, p_value = stats.ttest_ind(st_df['f1'], mt_df['f1'])

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((st_df['f1'].std()**2 + mt_df['f1'].std()**2) / 2)
        cohens_d = (mt_df['f1'].mean() - st_df['f1'].mean()) / pooled_std

        stats_results['t_test'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
        }

        stats_results['effect_size'] = {
            'cohens_d': float(cohens_d),
            'interpretation': (
                'large' if abs(cohens_d) > 0.8 else
                'medium' if abs(cohens_d) > 0.5 else
                'small'
            ),
        }

        stats_results['descriptive'] = {
            'single_task_mean_f1': float(st_df['f1'].mean()),
            'single_task_std_f1': float(st_df['f1'].std()),
            'multi_task_mean_f1': float(mt_df['f1'].mean()),
            'multi_task_std_f1': float(mt_df['f1'].std()),
        }

    return stats_results


def generate_summary_report(df: pd.DataFrame, stats: Dict) -> str:
    """Generate human-readable summary report."""

    report = []
    report.append("=" * 80)
    report.append("EXPERIMENT RESULTS SUMMARY")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    report.append(f"Total experiments: {len(df)}")
    report.append(f"Experiment types: {df['experiment_type'].unique().tolist()}")
    report.append(f"Models tested: {df['model'].unique().tolist()}")
    report.append(f"Datasets used: {df['dataset'].unique().tolist()}")
    report.append("")

    # Best results
    report.append("BEST RESULTS:")
    report.append("-" * 80)
    best = df.nlargest(5, 'f1')[['experiment_type', 'model', 'dataset', 'f1', 'precision', 'recall']]
    report.append(best.to_string())
    report.append("")

    # Statistical comparison
    if 'descriptive' in stats:
        report.append("STATISTICAL COMPARISON:")
        report.append("-" * 80)
        desc = stats['descriptive']
        report.append(f"Single-Task Mean F1: {desc['single_task_mean_f1']:.4f} ± {desc['single_task_std_f1']:.4f}")
        report.append(f"Multi-Task Mean F1:  {desc['multi_task_mean_f1']:.4f} ± {desc['multi_task_std_f1']:.4f}")
        report.append("")

        if 't_test' in stats:
            ttest = stats['t_test']
            report.append(f"T-test: t = {ttest['t_statistic']:.3f}, p = {ttest['p_value']:.4f}")
            report.append(f"Significant: {'YES' if ttest['significant'] else 'NO'} (α = 0.05)")

        if 'effect_size' in stats:
            effect = stats['effect_size']
            report.append(f"Cohen's d: {effect['cohens_d']:.3f} ({effect['interpretation']})")
        report.append("")

    # Token efficiency (RQ5)
    if 'tokens' in df.columns:
        report.append("TOKEN EFFICIENCY (RQ5):")
        report.append("-" * 80)

        for exp_type in df['experiment_type'].unique():
            exp_df = df[df['experiment_type'] == exp_type]
            if len(exp_df) > 0:
                avg_tokens = exp_df['tokens'].mean()
                avg_f1 = exp_df['f1'].mean()
                if avg_tokens > 0:
                    efficiency = avg_f1 / (avg_tokens / 1e6)  # F1 per million tokens
                    report.append(f"{exp_type:30s}: {avg_tokens:12,.0f} tokens → F1 {avg_f1:.4f} (eff: {efficiency:.4f})")
        report.append("")

    # Model comparison
    report.append("MODEL COMPARISON:")
    report.append("-" * 80)
    model_stats = df.groupby('model').agg({
        'f1': ['mean', 'std', 'count'],
        'trainable_params': 'first',
    }).round(4)
    report.append(model_stats.to_string())
    report.append("")

    report.append("=" * 80)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument('--results_dir', type=str, default='results/',
                       help='Directory containing result JSON files')
    parser.add_argument('--output_dir', type=str, default='analysis/',
                       help='Directory to save analysis outputs')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_all_results(results_dir)

    if len(results) == 0:
        print(f"No results found in {results_dir}")
        print("Make sure you have downloaded Kaggle experiment results.")
        return

    print(f"Loaded {len(results)} experiments")

    # Create comparison DataFrame
    df = create_comparison_dataframe(results)

    # Save full DataFrame
    df.to_csv(output_dir / 'all_results.csv', index=False)
    print(f"Saved: {output_dir / 'all_results.csv'}")

    # Generate tables
    print("\nGenerating LaTeX tables...")

    # Table 1: Single-task baselines
    st_table = generate_single_task_table(df)
    with open(output_dir / 'table_single_task.tex', 'w') as f:
        f.write(st_table)
    print(f"Saved: {output_dir / 'table_single_task.tex'}")

    # Table 2: Multi-task comparison
    mt_table = generate_multitask_comparison(df)
    with open(output_dir / 'table_multitask_comparison.tex', 'w') as f:
        f.write(mt_table)
    print(f"Saved: {output_dir / 'table_multitask_comparison.tex'}")

    # Table 3: Token-controlled analysis (RQ5 - CRITICAL!)
    tc_table = generate_token_controlled_analysis(df)
    with open(output_dir / 'table_rq5_token_controlled.tex', 'w') as f:
        f.write(tc_table)
    print(f"Saved: {output_dir / 'table_rq5_token_controlled.tex'}")

    # Compute statistics
    print("\nComputing statistical tests...")
    stats = compute_statistics(df)

    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {output_dir / 'statistics.json'}")

    # Generate summary report
    summary = generate_summary_report(df, stats)
    with open(output_dir / 'SUMMARY.txt', 'w') as f:
        f.write(summary)
    print(f"Saved: {output_dir / 'SUMMARY.txt'}")

    # Print summary to console
    print("\n" + summary)

    print(f"\n✅ Analysis complete! All outputs in {output_dir}/")
    print("\nFor your paper:")
    print("  - Copy LaTeX tables from .tex files")
    print("  - Use statistics.json for reporting p-values and effect sizes")
    print("  - Include SUMMARY.txt in supplementary materials")


if __name__ == '__main__':
    main()
