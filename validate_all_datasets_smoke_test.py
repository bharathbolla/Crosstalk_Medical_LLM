"""
Validate All 8 Datasets - Smoke Test
Tests one model (BioBERT) on all 8 tasks to verify everything works

Usage: python validate_all_datasets_smoke_test.py

Time: ~20-30 minutes (8 datasets × 3 min each)
Expected: All datasets train without errors
"""

import sys
import io
import torch
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer
)
import gc

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load complete implementation
print("Loading implementation...")
exec(open('COMPLETE_FIXED_DATASET.py').read())
exec(open('COMPLETE_FIXED_MODEL.py').read())
exec(open('COMPLETE_FIXED_METRICS.py').read())
print("✅ Implementation loaded\n")


def test_single_dataset(dataset_name, model_name='dmis-lab/biobert-v1.1', max_samples=100, num_epochs=3):
    """Test a single dataset with smoke test config."""

    print(f'\n{"="*60}')
    print(f'Testing: {dataset_name.upper()}')
    print(f'{"="*60}')

    try:
        # Load data
        pkl_file = Path('data/pickle') / f'{dataset_name}.pkl'
        if not pkl_file.exists():
            raise FileNotFoundError(f"Dataset not found: {pkl_file}")

        with open(pkl_file, 'rb') as f:
            raw_data = pickle.load(f)

        # Limit samples
        train_data = raw_data['train'][:max_samples]

        # Handle different split names
        val_split = 'validation' if 'validation' in raw_data else 'test'
        if val_split not in raw_data:
            # For datasets with only train (like pubmedqa)
            val_data = raw_data['train'][-20:]  # Last 20 samples
        else:
            val_data = raw_data[val_split][:max_samples//5]

        print(f'Dataset: {dataset_name}')
        print(f'Task type: {TASK_CONFIGS[dataset_name]["task_type"]}')
        print(f'Train samples: {len(train_data)}')
        print(f'Val samples: {len(val_data)}')

        # Load tokenizer
        if 'roberta' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create datasets
        train_dataset = UniversalMedicalDataset(
            data=train_data,
            tokenizer=tokenizer,
            task_name=dataset_name,
            max_length=256
        )

        val_dataset = UniversalMedicalDataset(
            data=val_data,
            tokenizer=tokenizer,
            task_name=dataset_name,
            max_length=256
        )

        # Get task config
        task_config = TASK_CONFIGS[dataset_name]

        # Prepare dataset stats
        dataset_stats = {
            dataset_name: {
                'task_type': task_config['task_type'],
                'model_type': task_config['model_type'],
                'num_labels': len(task_config['labels']) if task_config['labels'] else 1,
                'train_size': len(train_dataset),
                'val_size': len(val_dataset),
            }
        }

        # Load model
        print(f'\n🤖 Loading model...')
        model = load_model_for_task(model_name, dataset_name, dataset_stats)

        if torch.cuda.is_available():
            model = model.cuda()
            print('   ✅ Model on GPU')

        # Training args
        training_args = TrainingArguments(
            output_dir=f'./temp_checkpoints/{dataset_name}',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_steps=50,
            weight_decay=0.01,
            logging_steps=20,
            eval_strategy='epoch',
            save_strategy='no',  # Don't save to save space
            report_to='none',
            disable_tqdm=False,
        )

        # Setup metrics function
        if task_config['task_type'] == 'ner':
            def compute_metrics_fn(eval_pred):
                predictions = eval_pred.predictions
                labels = eval_pred.label_ids
                return compute_ner_metrics(predictions, labels, task_config['labels'])
        elif task_config['task_type'] in ['re', 'classification', 'qa']:
            def compute_metrics_fn(eval_pred):
                predictions = eval_pred.predictions
                labels = eval_pred.label_ids
                return compute_classification_metrics(predictions, labels)
        elif task_config['task_type'] == 'multilabel_classification':
            def compute_metrics_fn(eval_pred):
                predictions = eval_pred.predictions
                labels = eval_pred.label_ids
                return compute_multilabel_metrics(predictions, labels)
        elif task_config['task_type'] == 'similarity':
            def compute_metrics_fn(eval_pred):
                predictions = eval_pred.predictions
                labels = eval_pred.label_ids
                return compute_regression_metrics(predictions, labels)

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
        )

        print(f'\n🚀 Training {dataset_name} ({num_epochs} epochs)...')
        start_time = datetime.now()

        # Train
        train_result = trainer.train()

        elapsed = (datetime.now() - start_time).total_seconds()

        # Evaluate
        print(f'\n📊 Evaluating...')
        eval_result = trainer.evaluate()

        # Collect results
        result = {
            'dataset': dataset_name,
            'task_type': task_config['task_type'],
            'model_type': task_config['model_type'],
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'num_epochs': num_epochs,
            'f1': eval_result.get('eval_f1', 0),
            'precision': eval_result.get('eval_precision', 0),
            'recall': eval_result.get('eval_recall', 0),
            'accuracy': eval_result.get('eval_accuracy', 0),
            'train_loss': train_result.training_loss,
            'eval_loss': eval_result.get('eval_loss', 0),
            'time_seconds': elapsed,
            'status': 'PASS' if 'eval_f1' in eval_result and eval_result['eval_f1'] > 0.30 else 'PASS_OTHER'
        }

        # Handle regression (BIOSSES)
        if 'eval_pearson' in eval_result:
            result['pearson'] = eval_result['eval_pearson']
            result['mse'] = eval_result.get('eval_mse', 0)
            result['status'] = 'PASS' if result['pearson'] > 0.50 else 'FAIL'

        print(f'\n{"="*60}')
        print(f'✅ {dataset_name.upper()} COMPLETE')
        print(f'{"="*60}')
        print(f'Task type: {task_config["task_type"]}')
        if 'f1' in result and result['f1'] > 0:
            print(f'F1: {result["f1"]:.4f}')
            print(f'Precision: {result["precision"]:.4f}')
            print(f'Recall: {result["recall"]:.4f}')
        if 'pearson' in result:
            print(f'Pearson: {result["pearson"]:.4f}')
        print(f'Time: {elapsed:.1f}s')
        print(f'Status: {result["status"]}')

        # Cleanup
        del model, trainer, train_dataset, val_dataset, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return result

    except Exception as e:
        print(f'\n❌ ERROR with {dataset_name}:')
        print(f'   {str(e)}')
        import traceback
        traceback.print_exc()

        return {
            'dataset': dataset_name,
            'task_type': TASK_CONFIGS.get(dataset_name, {}).get('task_type', 'unknown'),
            'model_type': TASK_CONFIGS.get(dataset_name, {}).get('model_type', 'unknown'),
            'train_samples': 0,
            'val_samples': 0,
            'num_epochs': num_epochs,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0,
            'train_loss': 0.0,
            'eval_loss': 0.0,
            'time_seconds': 0,
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    """Test all 8 datasets."""

    print("="*60)
    print("COMPREHENSIVE DATASET VALIDATION")
    print("="*60)
    print("\nTesting BioBERT on all 8 datasets")
    print("Config: 100 samples, 3 epochs per dataset")
    print("Time: ~20-30 minutes total")
    print("="*60)

    # All 8 datasets
    datasets = [
        'bc2gm',       # NER
        'jnlpba',      # NER
        'chemprot',    # RE
        'ddi',         # RE
        'gad',         # Classification
        'hoc',         # Multi-label
        'pubmedqa',    # QA
        'biosses',     # Similarity
    ]

    results = []
    total_start = datetime.now()

    for i, dataset in enumerate(datasets, 1):
        print(f'\n\n{"#"*60}')
        print(f'DATASET {i}/{len(datasets)}')
        print(f'{"#"*60}')

        result = test_single_dataset(
            dataset_name=dataset,
            model_name='dmis-lab/biobert-v1.1',
            max_samples=100,
            num_epochs=3
        )
        results.append(result)

    total_elapsed = (datetime.now() - total_start).total_seconds()

    # Summary
    print('\n\n' + '='*60)
    print('📊 DATASET VALIDATION SUMMARY')
    print('='*60)

    df = pd.DataFrame(results)

    # Display results
    print('\nResults by Task Type:')
    print(df[['dataset', 'task_type', 'f1', 'precision', 'recall', 'time_seconds', 'status']].to_string(index=False))

    # Count by status
    passed = sum(1 for r in results if 'PASS' in r['status'])
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')

    print(f'\n{"="*60}')
    print(f'Passed: {passed}/{len(results)} ✅')
    print(f'Failed: {failed}/{len(results)} ❌')
    print(f'Errors: {errors}/{len(results)} 💥')
    print(f'Total time: {total_elapsed/60:.1f} minutes')
    print(f'{"="*60}')

    # Group by task type
    print('\n📊 Results by Task Type:')
    for task_type in df['task_type'].unique():
        task_df = df[df['task_type'] == task_type]
        avg_f1 = task_df['f1'].mean()
        print(f'  {task_type:25s} | Avg F1: {avg_f1:.4f} | Datasets: {len(task_df)}')

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = Path('results') / f'dataset_validation_{timestamp}.csv'
    results_file.parent.mkdir(exist_ok=True)
    df.to_csv(results_file, index=False)
    print(f'\n💾 Results saved: {results_file}')

    # Final verdict
    if failed == 0 and errors == 0:
        print('\n🎉 ALL DATASETS PASSED!')
        print('✅ Ready for full experiments (7 models × 8 tasks)')
        print('\nNext steps:')
        print('  1. Review results CSV')
        print('  2. Upload KAGGLE_COMPLETE.ipynb')
        print('  3. Set SMOKE_TEST = False')
        print('  4. Run full experiments!')
        return 0
    else:
        print(f'\n⚠️  {failed + errors} datasets had issues')
        print('Review errors above and check TROUBLESHOOTING_GUIDE.md')
        return 1


if __name__ == '__main__':
    sys.exit(main())
