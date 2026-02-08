"""
Notebook Integration: Test All 8 Datasets
Insert these cells into KAGGLE_COMPLETE.ipynb

Cell 4 (Modified): Configure multi-dataset test
Cell 10 (Modified): Loop through all datasets
"""

# ============================================
# CELL 4: Multi-Dataset Smoke Test Configuration
# ============================================

CELL_4_MULTI_DATASET = '''
# ============================================
# 🔥 MULTI-DATASET SMOKE TEST
# ============================================

import sys
import io
if sys.platform != 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('\\n' + '='*60)
print('🔥 MULTI-DATASET SMOKE TEST')
print('='*60)

# ⭐ Set to True to test all 8 datasets
TEST_ALL_DATASETS = True  # Change to False for single dataset

if TEST_ALL_DATASETS:
    print('\\n✅ TESTING ALL 8 DATASETS')

    # All 8 datasets
    all_datasets = [
        'bc2gm',       # NER
        'jnlpba',      # NER
        'chemprot',    # RE
        'ddi',         # RE
        'gad',         # Classification
        'hoc',         # Multi-label
        'pubmedqa',    # QA
        'biosses',     # Similarity
    ]

    # Store for later
    CONFIG['all_datasets_to_test'] = all_datasets
    CONFIG['dataset_test_results'] = []

    # Smoke test config
    CONFIG['max_samples_per_dataset'] = 100  # 100 samples per dataset
    CONFIG['num_epochs'] = 3  # 3 epochs (quick test)
    CONFIG['batch_size'] = 16
    CONFIG['max_length'] = 256
    CONFIG['warmup_steps'] = 50
    CONFIG['use_early_stopping'] = False

    print(f'\\nDatasets to test: {len(all_datasets)}')
    for i, ds in enumerate(all_datasets, 1):
        task_type = TASK_CONFIGS[ds]['task_type']
        print(f'  {i}. {ds:12s} | {task_type}')

    print('\\n⏱️  Expected time: ~20-30 minutes')
    print('📊 Expected: All datasets train successfully')

else:
    print('\\n✅ SINGLE DATASET TEST')
    # Single dataset test (original behavior)
    CONFIG['max_samples_per_dataset'] = 100
    CONFIG['num_epochs'] = 3
    CONFIG['batch_size'] = 16
    CONFIG['max_length'] = 256

    print(f"   Dataset: {CONFIG['datasets'][0]}")
    print('   Settings: 100 samples, 3 epochs')

print('='*60)
'''

# ============================================
# CELL 10: Multi-Dataset Training Loop
# Replace Cell 10 with this when TEST_ALL_DATASETS = True
# ============================================

CELL_10_MULTI_DATASET = '''
# ============================================
# MULTI-DATASET TRAINING LOOP
# ============================================

import torch
import gc
import pandas as pd
from datetime import datetime

if 'all_datasets_to_test' in CONFIG:
    print('\\n' + '='*60)
    print('🚀 MULTI-DATASET TRAINING LOOP')
    print('='*60)

    all_results = []

    for ds_idx, dataset_to_test in enumerate(CONFIG['all_datasets_to_test'], 1):
        print(f'\\n{"="*60}')
        print(f'DATASET {ds_idx}/{len(CONFIG["all_datasets_to_test"])}: {dataset_to_test.upper()}')
        print(f'{"="*60}')

        try:
            # Update current dataset
            primary_dataset = dataset_to_test
            task_config = TASK_CONFIGS[primary_dataset]

            print(f'Task type: {task_config["task_type"]}')
            print(f'Model type: {task_config["model_type"]}')

            # Step 1: Load data
            print(f'\\n📊 Loading {primary_dataset} data...')
            pickle_file = Path('data/pickle') / f'{primary_dataset}.pkl'

            with open(pickle_file, 'rb') as f:
                raw_data = pickle.load(f)

            # Handle different splits
            val_split = 'validation' if 'validation' in raw_data else 'test'
            if val_split not in raw_data:
                raw_data[val_split] = raw_data['train'][-20:]

            # Limit samples
            if CONFIG['max_samples_per_dataset']:
                raw_data['train'] = raw_data['train'][:CONFIG['max_samples_per_dataset']]
                raw_data[val_split] = raw_data[val_split][:CONFIG['max_samples_per_dataset']//5]

            print(f"   Train samples: {len(raw_data['train']):,}")
            print(f"   Val samples: {len(raw_data[val_split]):,}")

            # Step 2: Load tokenizer
            print(f'\\n🔤 Loading tokenizer...')
            model_name = CONFIG['model_name']

            if 'roberta' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Step 3: Create datasets
            print(f'\\n📦 Creating datasets...')
            train_dataset = UniversalMedicalDataset(
                data=raw_data['train'],
                tokenizer=tokenizer,
                task_name=primary_dataset,
                max_length=CONFIG['max_length']
            )

            val_dataset = UniversalMedicalDataset(
                data=raw_data[val_split],
                tokenizer=tokenizer,
                task_name=primary_dataset,
                max_length=CONFIG['max_length']
            )

            # Prepare dataset stats
            dataset_stats = {
                primary_dataset: {
                    'task_type': task_config['task_type'],
                    'model_type': task_config['model_type'],
                    'num_labels': len(task_config['labels']) if task_config['labels'] else 1,
                    'train_size': len(train_dataset),
                    'val_size': len(val_dataset),
                }
            }

            print(f'   ✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}')

            # Step 4: Load model
            print(f'\\n🤖 Loading model...')
            model = load_model_for_task(model_name, primary_dataset, dataset_stats)

            if torch.cuda.is_available():
                model = model.cuda()
                print('   ✅ Model on GPU')

            # Step 5: Setup trainer
            print(f'\\n⚙️  Setting up trainer...')
            training_args = TrainingArguments(
                output_dir=f"./checkpoints/{primary_dataset}_{EXPERIMENT_ID}",
                num_train_epochs=CONFIG['num_epochs'],
                per_device_train_batch_size=CONFIG['batch_size'],
                per_device_eval_batch_size=CONFIG['batch_size'],
                learning_rate=CONFIG['learning_rate'],
                warmup_steps=CONFIG['warmup_steps'],
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=20,
                eval_strategy='epoch',
                save_strategy='no',  # Don't save to save disk space
                report_to='none',
                disable_tqdm=False,
            )

            # Setup metrics
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

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics_fn,
            )

            print('   ✅ Trainer ready')

            # Step 6: Train!
            print(f'\\n🚀 Training {primary_dataset}...')
            start_time = datetime.now()

            train_result = trainer.train()

            elapsed = (datetime.now() - start_time).total_seconds()

            # Step 7: Evaluate
            print(f'\\n📊 Evaluating...')
            eval_result = trainer.evaluate()

            # Collect results
            result = {
                'dataset': primary_dataset,
                'task_type': task_config['task_type'],
                'model_type': task_config['model_type'],
                'train_samples': len(raw_data['train']),
                'val_samples': len(raw_data[val_split]),
                'num_epochs': CONFIG['num_epochs'],
                'f1': eval_result.get('eval_f1', 0),
                'precision': eval_result.get('eval_precision', 0),
                'recall': eval_result.get('eval_recall', 0),
                'accuracy': eval_result.get('eval_accuracy', 0),
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result.get('eval_loss', 0),
                'time_seconds': elapsed,
            }

            # Handle regression (BIOSSES)
            if 'eval_pearson' in eval_result:
                result['pearson'] = eval_result['eval_pearson']
                result['mse'] = eval_result.get('eval_mse', 0)

            # Determine status
            if result['f1'] > 0.30:
                result['status'] = 'PASS'
            elif 'pearson' in result and result['pearson'] > 0.50:
                result['status'] = 'PASS'
            else:
                result['status'] = 'FAIL'

            all_results.append(result)

            # Print result
            print(f'\\n{"="*60}')
            print(f'✅ {primary_dataset.upper()} COMPLETE')
            print(f'{"="*60}')
            if result['f1'] > 0:
                print(f'F1: {result["f1"]:.4f}')
                print(f'Precision: {result["precision"]:.4f}')
                print(f'Recall: {result["recall"]:.4f}')
            if 'pearson' in result:
                print(f'Pearson: {result["pearson"]:.4f}')
            print(f'Time: {elapsed:.1f}s')
            print(f'Status: {result["status"]} {"✅" if result["status"] == "PASS" else "❌"}')

            # Clean up GPU memory
            del model, trainer, train_dataset, val_dataset, tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f'\\n❌ ERROR with {dataset_to_test}:')
            print(f'   {str(e)}')
            import traceback
            traceback.print_exc()

            # Record error
            all_results.append({
                'dataset': dataset_to_test,
                'task_type': TASK_CONFIGS.get(dataset_to_test, {}).get('task_type', 'unknown'),
                'model_type': TASK_CONFIGS.get(dataset_to_test, {}).get('model_type', 'unknown'),
                'train_samples': 0,
                'val_samples': 0,
                'num_epochs': CONFIG['num_epochs'],
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'accuracy': 0.0,
                'train_loss': 0.0,
                'eval_loss': 0.0,
                'time_seconds': 0,
                'status': 'ERROR',
                'error': str(e)
            })

            # Continue to next dataset
            continue

    # ============================================
    # SUMMARY OF ALL RESULTS
    # ============================================

    print('\\n' + '='*60)
    print('📊 MULTI-DATASET VALIDATION SUMMARY')
    print('='*60)

    # Create DataFrame
    df_results = pd.DataFrame(all_results)

    # Display results
    print('\\nResults:')
    print(df_results[['dataset', 'task_type', 'f1', 'precision', 'recall', 'time_seconds', 'status']].to_string(index=False))

    # Count by status
    passed = sum(1 for r in all_results if r['status'] == 'PASS')
    failed = sum(1 for r in all_results if r['status'] == 'FAIL')
    errors = sum(1 for r in all_results if r['status'] == 'ERROR')

    print(f'\\n{"="*60}')
    print(f'Passed: {passed}/{len(all_results)} ✅')
    print(f'Failed: {failed}/{len(all_results)} ❌')
    print(f'Errors: {errors}/{len(all_results)} 💥')
    print(f'{"="*60}')

    # Group by task type
    print('\\n📊 Results by Task Type:')
    for task_type in df_results['task_type'].unique():
        task_df = df_results[df_results['task_type'] == task_type]
        avg_f1 = task_df['f1'].mean()
        count = len(task_df)
        print(f'  {task_type:25s} | Avg F1: {avg_f1:.4f} | Datasets: {count}')

    # Save results
    results_file = RESULTS_DIR / f'dataset_validation_{EXPERIMENT_ID}.csv'
    df_results.to_csv(results_file, index=False)
    print(f'\\n💾 Results saved to: {results_file}')

    # Final verdict
    if failed == 0 and errors == 0:
        print('\\n🎉 ALL DATASETS PASSED!')
        print('✅ Ready for full experiments (7 models × 8 tasks)')
    else:
        print(f'\\n⚠️  {failed + errors} datasets had issues')
        print('Review errors above')

    # Store results
    CONFIG['dataset_test_results'] = all_results

else:
    print('\\n⚠️  TEST_ALL_DATASETS not enabled')
    print('Run Cell 4 with TEST_ALL_DATASETS = True first')
'''

# ============================================
# How to use in notebook
# ============================================

USAGE_INSTRUCTIONS = """
USAGE INSTRUCTIONS:

1. Open KAGGLE_COMPLETE.ipynb in Kaggle

2. REPLACE Cell 4 with content from CELL_4_MULTI_DATASET above
   - This configures multi-dataset testing
   - Set TEST_ALL_DATASETS = True

3. REPLACE Cell 10 with content from CELL_10_MULTI_DATASET above
   - This loops through all 8 datasets
   - Trains and evaluates each one
   - Shows summary at the end

4. Run Cells 1-10
   - Wait ~20-30 minutes
   - Check summary table
   - Review results CSV

5. Expected Results:
   - BC2GM (NER): F1 > 0.60
   - JNLPBA (NER): F1 > 0.55
   - ChemProt (RE): F1 > 0.55
   - DDI (RE): F1 > 0.60
   - GAD (Class): F1 > 0.70
   - HoC (Multi): F1 > 0.40
   - PubMedQA (QA): F1 > 0.50
   - BIOSSES (Sim): Pearson > 0.60

6. If all pass:
   ✅ Proceed with full experiments (7 models × 8 tasks)
   ✅ Set TEST_ALL_DATASETS = False
   ✅ Run complete matrix

7. If some fail:
   ❌ Check error messages
   ❌ Review TROUBLESHOOTING_GUIDE.md
   ❌ Fix issues and re-run
"""

if __name__ == '__main__':
    print("="*60)
    print("MULTI-DATASET NOTEBOOK INTEGRATION")
    print("="*60)
    print(USAGE_INSTRUCTIONS)
    print("="*60)
    print("\\nCell contents:")
    print("\\n1. CELL_4_MULTI_DATASET - Replace Cell 4 with this")
    print("2. CELL_10_MULTI_DATASET - Replace Cell 10 with this")
    print("="*60)
