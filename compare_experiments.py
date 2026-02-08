"""
Compare Experiments - Quick Analysis Tool

Reads the master CSV file (all_experiments.csv) and generates comparison tables
for your research paper.

Usage:
    python compare_experiments.py results/all_experiments.csv
"""

import pandas as pd
import sys
from pathlib import Path


def load_results(csv_path):
    """Load results from CSV file."""
    df = pd.read_csv(csv_path)
    return df


def compare_by_model(df):
    """Compare performance across different models."""
    print("\n" + "="*80)
    print("COMPARISON BY MODEL")
    print("="*80)

    # Group by model and dataset
    comparison = df.groupby(['model_name', 'dataset']).agg({
        'test_f1': ['mean', 'std', 'count'],
        'test_precision': 'mean',
        'test_recall': 'mean',
        'tokens_processed': 'mean',
        'train_runtime_seconds': 'mean',
    }).round(4)

    print(comparison)
    return comparison


def compare_by_dataset(df):
    """Compare performance across different datasets."""
    print("\n" + "="*80)
    print("COMPARISON BY DATASET")
    print("="*80)

    comparison = df.groupby('dataset').agg({
        'test_f1': ['mean', 'std', 'min', 'max'],
        'test_precision': 'mean',
        'test_recall': 'mean',
        'model_name': 'count',
    }).round(4)

    print(comparison)
    return comparison


def compare_single_vs_multi_task(df):
    """Compare single-task vs multi-task experiments."""
    print("\n" + "="*80)
    print("SINGLE-TASK vs MULTI-TASK COMPARISON")
    print("="*80)

    comparison = df.groupby('experiment_type').agg({
        'test_f1': ['mean', 'std'],
        'test_precision': 'mean',
        'test_recall': 'mean',
        'tokens_processed': 'mean',
        'train_runtime_seconds': 'mean',
        'experiment_id': 'count',
    }).round(4)

    print(comparison)
    return comparison


def find_best_models(df, top_n=5):
    """Find top N models by F1 score."""
    print("\n" + "="*80)
    print(f"TOP {top_n} EXPERIMENTS BY F1 SCORE")
    print("="*80)

    top_models = df.nlargest(top_n, 'test_f1')[
        ['experiment_id', 'model_name', 'dataset', 'experiment_type',
         'test_f1', 'test_precision', 'test_recall', 'tokens_processed']
    ]

    print(top_models.to_string(index=False))
    return top_models


def token_efficiency_analysis(df):
    """Analyze F1 score per million tokens (efficiency)."""
    print("\n" + "="*80)
    print("TOKEN EFFICIENCY ANALYSIS (F1 per Million Tokens)")
    print("="*80)

    df['f1_per_million_tokens'] = (df['test_f1'] / (df['tokens_processed'] / 1_000_000)).round(4)

    efficiency = df.groupby('model_name').agg({
        'f1_per_million_tokens': ['mean', 'std'],
        'test_f1': 'mean',
        'tokens_processed': 'mean',
    }).round(4)

    print(efficiency)
    return efficiency


def generate_latex_table(df, output_file='results/latex_table.tex'):
    """Generate LaTeX table for paper."""
    print("\n" + "="*80)
    print("GENERATING LATEX TABLE")
    print("="*80)

    # Create summary table
    summary = df.groupby(['experiment_type', 'model_name', 'dataset']).agg({
        'test_f1': 'mean',
        'test_precision': 'mean',
        'test_recall': 'mean',
    }).round(3)

    latex = summary.to_latex()

    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"✅ LaTeX table saved to: {output_file}")
    print("\nPreview:")
    print(summary)

    return latex


def training_efficiency_analysis(df):
    """Analyze training efficiency (samples per second, runtime)."""
    print("\n" + "="*80)
    print("TRAINING EFFICIENCY ANALYSIS")
    print("="*80)

    efficiency = df.groupby(['model_name', 'batch_size']).agg({
        'train_samples_per_second': ['mean', 'std'],
        'train_runtime_seconds': 'mean',
        'test_f1': 'mean',
    }).round(2)

    print(efficiency)
    return efficiency


def early_stopping_analysis(df):
    """Analyze early stopping behavior."""
    print("\n" + "="*80)
    print("EARLY STOPPING ANALYSIS")
    print("="*80)

    # Filter only early stopping experiments
    early_stop_df = df[df['early_stopping'] == True]

    if len(early_stop_df) > 0:
        analysis = early_stop_df.groupby('dataset').agg({
            'actual_epochs': ['mean', 'min', 'max'],
            'num_epochs_max': 'first',
            'test_f1': 'mean',
        }).round(2)

        print(analysis)
        print(f"\nAverage epochs used: {early_stop_df['actual_epochs'].mean():.2f}")
        print(f"Max allowed: {early_stop_df['num_epochs_max'].iloc[0]}")
        print(f"Epochs saved: {early_stop_df['num_epochs_max'].iloc[0] - early_stop_df['actual_epochs'].mean():.2f}")
    else:
        print("No early stopping experiments found.")

    return early_stop_df if len(early_stop_df) > 0 else None


def main():
    """Main analysis function."""
    if len(sys.argv) < 2:
        csv_file = "results/all_experiments.csv"
    else:
        csv_file = sys.argv[1]

    csv_path = Path(csv_file)

    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        print(f"\nExpected location: {csv_path.absolute()}")
        print("\nMake sure you've run at least one experiment!")
        return

    print("="*80)
    print("EXPERIMENT COMPARISON TOOL")
    print("="*80)
    print(f"Loading: {csv_path}")

    df = load_results(csv_path)

    print(f"\n✅ Loaded {len(df)} experiments")
    print(f"   Models: {df['model_name'].nunique()}")
    print(f"   Datasets: {df['dataset'].nunique()}")
    print(f"   Experiment types: {df['experiment_type'].nunique()}")

    # Run all analyses
    compare_by_model(df)
    compare_by_dataset(df)
    compare_single_vs_multi_task(df)
    find_best_models(df, top_n=5)
    token_efficiency_analysis(df)
    training_efficiency_analysis(df)
    early_stopping_analysis(df)
    generate_latex_table(df)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults CSV: {csv_path}")
    print(f"LaTeX table: results/latex_table.tex")
    print(f"\nTo re-run analysis: python compare_experiments.py {csv_file}")


if __name__ == "__main__":
    main()
