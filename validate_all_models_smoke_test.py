"""
Validate All 7 Models - Comprehensive Smoke Test
Run this locally or on Kaggle to test all models before full experiments

Usage: python validate_all_models_smoke_test.py

Time: ~15 minutes (2 min per model)
Expected: All models F1 > 0.30 on BC2GM
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
    AutoConfig,
    TrainingArguments,
    Trainer
)

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load complete implementation
exec(open('COMPLETE_FIXED_DATASET.py').read())
exec(open('COMPLETE_FIXED_MODEL.py').read())
exec(open('COMPLETE_FIXED_METRICS.py').read())


def test_single_model(model_name, dataset_name='bc2gm', max_samples=50):
    """Test a single model with smoke test config."""

    print(f'\n{"="*60}')
    print(f'Testing: {model_name}')
    print(f'{"="*60}')

    try:
        # Load data
        pkl_file = Path('data/pickle') / f'{dataset_name}.pkl'
        with open(pkl_file, 'rb') as f:
            raw_data = pickle.load(f)

        # Limit samples
        train_data = raw_data['train'][:max_samples]
        val_split = 'validation' if 'validation' in raw_data else 'test'
        val_data = raw_data[val_split][:max_samples//5]

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
            max_length=128
        )

        val_dataset = UniversalMedicalDataset(
            data=val_data,
            tokenizer=tokenizer,
            task_name=dataset_name,
            max_length=128
        )

        # Load model
        task_config = TASK_CONFIGS[dataset_name]
        dataset_stats = {
            dataset_name: {
                'task_type': task_config['task_type'],
                'model_type': task_config['model_type'],
                'num_labels': len(task_config['labels']) if task_config['labels'] else 1,
                'train_size': len(train_dataset),
                'val_size': len(val_dataset),
            }
        }

        model = load_model_for_task(model_name, dataset_name, dataset_stats)

        if torch.cuda.is_available():
            model = model.cuda()

        # Training args
        training_args = TrainingArguments(
            output_dir=f'./temp_checkpoints/{model_name.split("/")[-1]}',
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            warmup_steps=10,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='no',  # Don't save checkpoints
            report_to='none',
            disable_tqdm=False,
        )

        # Metrics function
        if task_config['task_type'] == 'ner':
            def compute_metrics_fn(eval_pred):
                predictions = eval_pred.predictions
                labels = eval_pred.label_ids
                return compute_ner_metrics(predictions, labels, task_config['labels'])
        else:
            def compute_metrics_fn(eval_pred):
                predictions = eval_pred.predictions
                labels = eval_pred.label_ids
                return compute_classification_metrics(predictions, labels)

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
        )

        # Train
        print('\n🚀 Training...')
        start_time = datetime.now()
        train_result = trainer.train()
        elapsed = (datetime.now() - start_time).total_seconds()

        # Evaluate
        print('📊 Evaluating...')
        eval_result = trainer.evaluate()

        # Results
        result = {
            'model': model_name,
            'model_short': model_name.split('/')[-1],
            'dataset': dataset_name,
            'f1': eval_result.get('eval_f1', 0),
            'precision': eval_result.get('eval_precision', 0),
            'recall': eval_result.get('eval_recall', 0),
            'train_loss': train_result.training_loss,
            'eval_loss': eval_result.get('eval_loss', 0),
            'time_seconds': elapsed,
            'status': 'PASS' if eval_result.get('eval_f1', 0) > 0.30 else 'FAIL'
        }

        print(f'\n✅ Complete: F1 = {result["f1"]:.4f} ({result["status"]})')

        # Cleanup
        del model, trainer, train_dataset, val_dataset, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f'\n❌ ERROR: {str(e)}')
        import traceback
        traceback.print_exc()

        return {
            'model': model_name,
            'model_short': model_name.split('/')[-1],
            'dataset': dataset_name,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'train_loss': 0.0,
            'eval_loss': 0.0,
            'time_seconds': 0,
            'status': 'ERROR',
            'error': str(e)
        }


def main():
    """Test all 7 models."""

    print("="*60)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("="*60)
    print("\nTesting all 7 BERT models on BC2GM")
    print("Smoke test: 50 samples, 1 epoch")
    print("Expected: All models F1 > 0.30")
    print("Time: ~15 minutes")
    print("="*60)

    # All 7 models
    models = [
        'dmis-lab/biobert-v1.1',
        'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        'allenai/biomed_roberta_base',
        'emilyalsentzer/Bio_ClinicalBERT',
        'roberta-base',
        'bert-base-uncased',
    ]

    results = []
    for i, model in enumerate(models, 1):
        print(f'\n\n{"#"*60}')
        print(f'MODEL {i}/{len(models)}')
        print(f'{"#"*60}')

        result = test_single_model(model, dataset_name='bc2gm', max_samples=50)
        results.append(result)

    # Summary
    print('\n\n' + '='*60)
    print('📊 VALIDATION SUMMARY')
    print('='*60)

    df = pd.DataFrame(results)
    print('\nResults:')
    print(df[['model_short', 'f1', 'precision', 'recall', 'time_seconds', 'status']].to_string(index=False))

    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')

    print(f'\n{"="*60}')
    print(f'Passed: {passed}/{len(results)} ✅')
    print(f'Failed: {failed}/{len(results)} ❌')
    print(f'Errors: {errors}/{len(results)} 💥')
    print(f'{"="*60}')

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = Path('results') / f'model_validation_{timestamp}.csv'
    results_file.parent.mkdir(exist_ok=True)
    df.to_csv(results_file, index=False)
    print(f'\n💾 Results saved: {results_file}')

    # Final verdict
    if failed == 0 and errors == 0:
        print('\n🎉 ALL MODELS PASSED!')
        print('✅ Ready for full experiments')
        return 0
    else:
        print(f'\n⚠️  {failed + errors} models had issues')
        print('Check results above')
        return 1


if __name__ == '__main__':
    sys.exit(main())
