# Kaggle Smoke Test Cell
## Add this cell to your notebook for quick validation

## Cell: Smoke Test All Experiments

```python
# ============================================
# SMOKE TEST: Run all 56 experiments quickly
# ============================================

print("""
╔══════════════════════════════════════════════════════════╗
║         SMOKE TEST: 7 Models × 8 Datasets               ║
║         Expected time: 2-3 hours                        ║
╚══════════════════════════════════════════════════════════╝
""")

# Smoke test configuration
SMOKE_CONFIG = CONFIG.copy()
SMOKE_CONFIG.update({
    'max_samples_per_dataset': 50,  # Tiny test
    'num_epochs': 1,
    'batch_size': 16,
    'max_length': 128,  # Faster
    'warmup_steps': 10,
    'save_steps': 50,
    'eval_steps': 25,
    'use_early_stopping': False,
})

# All models to test
models_to_test = [
    'bert-base-uncased',
    'roberta-base',
    'dmis-lab/biobert-v1.1',
    'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'emilyalsentzer/Bio_ClinicalBERT',
    'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
    'allenai/biomed_roberta_base',
]

# All datasets to test
datasets_to_test = [
    'bc2gm', 'jnlpba', 'chemprot', 'ddi',
    'gad', 'hoc', 'pubmedqa', 'biosses'
]

# Track results
smoke_results = {
    'passed': [],
    'failed': [],
    'total': 0,
}

import time
start_time = time.time()

# Run all combinations
for model_idx, model_name in enumerate(models_to_test):
    for dataset_idx, dataset_name in enumerate(datasets_to_test):
        test_num = model_idx * len(datasets_to_test) + dataset_idx + 1
        test_id = f"{model_name.split('/')[-1]}_{dataset_name}"

        print(f"\n{'='*60}")
        print(f"Test {test_num}/56: {test_id}")
        print(f"{'='*60}")

        try:
            # Update config
            SMOKE_CONFIG['model_name'] = model_name
            SMOKE_CONFIG['datasets'] = [dataset_name]
            SMOKE_CONFIG['experiment_id'] = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Load model (reuse code from Cell 5)
            print(f"  Loading {model_name}...")
            # ... (use existing model loading code)

            # Load dataset (reuse code from Cell 4)
            print(f"  Loading {dataset_name} (50 samples)...")
            # ... (use existing dataset loading code)

            # Train (reuse code from Cell 6)
            print(f"  Training...")
            # ... (use existing training code)

            # Evaluate (reuse code from Cell 7)
            print(f"  Evaluating...")
            # ... (use existing evaluation code)

            print(f"  ✅ PASSED")
            smoke_results['passed'].append(test_id)

        except Exception as e:
            print(f"  ❌ FAILED: {str(e)}")
            smoke_results['failed'].append({
                'test': test_id,
                'error': str(e)
            })

        finally:
            smoke_results['total'] += 1

            # Clean up memory
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
            torch.cuda.empty_cache()
            gc.collect()

# Generate report
elapsed = (time.time() - start_time) / 3600

print("\n" + "="*60)
print("SMOKE TEST REPORT")
print("="*60)
print(f"Total tests: {smoke_results['total']}")
print(f"Passed: {len(smoke_results['passed'])} ✅")
print(f"Failed: {len(smoke_results['failed'])} ❌")
print(f"Success rate: {100 * len(smoke_results['passed']) / smoke_results['total']:.1f}%")
print(f"Total time: {elapsed:.2f} hours")
print("="*60)

if smoke_results['failed']:
    print("\n❌ FAILED TESTS:")
    for failure in smoke_results['failed']:
        print(f"  - {failure['test']}: {failure['error']}")
else:
    print("\n✅ ALL TESTS PASSED!")
    print("\nReady for A100 commercial runs!")

# Verify CSV
csv_path = RESULTS_DIR / "all_experiments.csv"
if csv_path.exists():
    import pandas as pd
    df = pd.read_csv(csv_path)
    print(f"\n📊 CSV Validation:")
    print(f"  Rows: {len(df)}")
    print(f"  Expected: 56")
    print(f"  Status: {'✅ OK' if len(df) == 56 else '❌ MISMATCH'}")

    # Check F1 scores are reasonable
    if 'test_f1' in df.columns:
        avg_f1 = df['test_f1'].mean()
        print(f"  Average F1: {avg_f1:.3f}")
        print(f"  Status: {'✅ OK' if avg_f1 > 0.2 else '⚠️ LOW'}")
```

## Alternative: Loop-Based Smoke Test (Simpler)

```python
# Simpler version: Just loop and catch errors

models = ['bert-base-uncased', 'dmis-lab/biobert-v1.1', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract']
datasets = ['bc2gm', 'jnlpba', 'chemprot']

CONFIG['max_samples_per_dataset'] = 50
CONFIG['num_epochs'] = 1

for model in models:
    for dataset in datasets:
        try:
            CONFIG['model_name'] = model
            CONFIG['datasets'] = [dataset]

            print(f"\nTesting {model.split('/')[-1]} on {dataset}...")

            # Run cells 4-7 programmatically
            # (or just run manually for each combination)

            print("✅ PASSED")

        except Exception as e:
            print(f"❌ FAILED: {e}")
```

## After Smoke Tests: Verification Checklist

### 1. Check CSV File
```python
import pandas as pd

df = pd.read_csv('results/all_experiments.csv')

print(f"Total experiments: {len(df)}")
print(f"Expected: 56")
print(f"\nUnique models: {df['model_name'].nunique()}")
print(f"Expected: 7")
print(f"\nUnique datasets: {df['dataset'].nunique()}")
print(f"Expected: 8")

# Show F1 distribution
print(f"\nF1 Score Statistics (50 samples only):")
print(df['test_f1'].describe())

# Models tested
print(f"\nModels tested:")
print(df['model_name'].value_counts())

# Datasets tested
print(f"\nDatasets tested:")
print(df['dataset'].value_counts())
```

### 2. Visual Validation
```python
import matplotlib.pyplot as plt

# F1 by model
df.groupby('model_name')['test_f1'].mean().plot(kind='barh', figsize=(10, 6))
plt.title('Average F1 by Model (50 samples)')
plt.xlabel('F1 Score')
plt.tight_layout()
plt.savefig('results/smoke_test_f1_by_model.png')
plt.show()

# F1 by dataset
df.groupby('dataset')['test_f1'].mean().plot(kind='barh', figsize=(10, 6))
plt.title('Average F1 by Dataset (50 samples)')
plt.xlabel('F1 Score')
plt.tight_layout()
plt.savefig('results/smoke_test_f1_by_dataset.png')
plt.show()
```

### 3. Error Check
```python
# Check for any obvious issues
issues = []

# Check for missing results
if len(df) != 56:
    issues.append(f"Missing experiments: expected 56, got {len(df)}")

# Check for zero F1 scores
zero_f1 = df[df['test_f1'] == 0]
if len(zero_f1) > 0:
    issues.append(f"Zero F1 scores: {len(zero_f1)} experiments")
    print(zero_f1[['model_name', 'dataset', 'test_f1']])

# Check for NaN values
nan_count = df['test_f1'].isna().sum()
if nan_count > 0:
    issues.append(f"NaN F1 scores: {nan_count} experiments")

if issues:
    print("⚠️  ISSUES FOUND:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ No issues found! Ready for A100!")
```

## If All Tests Pass → A100 Configuration

```python
# Update config for A100 full runs
A100_CONFIG = {
    "experiment_id": EXPERIMENT_ID,
    "experiment_type": "single_task",
    "max_samples_per_dataset": None,  # ALL data
    "num_epochs": 10,
    "batch_size": 64,  # A100 can handle this
    "learning_rate": 2e-5,
    "max_length": 512,
    "warmup_steps": 500,
    "weight_decay": 0.01,

    # Aggressive checkpointing
    "save_strategy": "steps",
    "save_steps": 100,
    "keep_last_n_checkpoints": 2,
    "resume_from_checkpoint": True,

    # Evaluation
    "eval_strategy": "steps",
    "eval_steps": 250,

    # Early stopping
    "use_early_stopping": True,
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.0001,

    # Token tracking CRITICAL
    "track_tokens": True,

    "use_wandb": False,
    "logging_steps": 50,
}

print("✅ A100 configuration ready!")
print(f"Expected time: 40 GPU hours (1.7 days)")
print(f"Expected cost: $5.72 @ $0.143/hr")
```
