"""
Multi-Model Training Loop
Insert this as a new cell after Cell 9 (before Cell 10)
Loops through all 7 models and trains each
"""

# ============================================
# MULTI-MODEL TRAINING LOOP
# ============================================

import torch
import gc
import pandas as pd
from datetime import datetime

if 'all_models_to_test' in CONFIG:
    print('\n' + '='*60)
    print('🚀 MULTI-MODEL TRAINING LOOP')
    print('='*60)

    all_results = []

    for model_idx, model_name_to_test in enumerate(CONFIG['all_models_to_test'], 1):
        print(f'\n{"="*60}')
        print(f'MODEL {model_idx}/{len(CONFIG["all_models_to_test"])}: {model_name_to_test}')
        print(f'{"="*60}')

        try:
            # Update model name
            CONFIG['model_name'] = model_name_to_test
            model_short_name = model_name_to_test.split('/')[-1]

            # Step 1: Load tokenizer
            print(f'\n🔤 Loading tokenizer for {model_short_name}...')
            if 'roberta' in model_name_to_test.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name_to_test, add_prefix_space=True)
                print('   ✅ RoBERTa tokenizer (add_prefix_space=True)')
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name_to_test)
                print('   ✅ BERT tokenizer')

            # Step 2: Reload datasets with new tokenizer
            print(f'\n📊 Reloading datasets...')
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
            print(f'   ✅ Train: {len(train_dataset)}, Val: {len(val_dataset)}')

            # Step 3: Load model
            print(f'\n🤖 Loading model...')
            model = load_model_for_task(model_name_to_test, primary_dataset, dataset_stats)
            if torch.cuda.is_available():
                model = model.cuda()
            print('   ✅ Model loaded and on GPU')

            # Step 4: Setup trainer
            print(f'\n⚙️  Setting up trainer...')
            training_args = TrainingArguments(
                output_dir=f"./checkpoints/{model_short_name}_{primary_dataset}_{EXPERIMENT_ID}",
                num_train_epochs=CONFIG['num_epochs'],
                per_device_train_batch_size=CONFIG['batch_size'],
                per_device_eval_batch_size=CONFIG['batch_size'],
                learning_rate=CONFIG['learning_rate'],
                warmup_steps=CONFIG['warmup_steps'],
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=50,
                eval_strategy='epoch',
                save_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                greater_is_better=True,
                save_total_limit=1,  # Save space
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

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics_fn,
            )
            print('   ✅ Trainer ready')

            # Step 5: Train!
            print(f'\n🚀 Training {model_short_name}...')
            start_time = datetime.now()

            train_result = trainer.train()

            elapsed = (datetime.now() - start_time).total_seconds()

            # Step 6: Evaluate
            print(f'\n📊 Evaluating {model_short_name}...')
            eval_result = trainer.evaluate()

            # Collect results
            result = {
                'model': model_name_to_test,
                'model_short': model_short_name,
                'dataset': primary_dataset,
                'f1': eval_result.get('eval_f1', 0),
                'precision': eval_result.get('eval_precision', 0),
                'recall': eval_result.get('eval_recall', 0),
                'train_loss': train_result.training_loss,
                'eval_loss': eval_result.get('eval_loss', 0),
                'time_seconds': elapsed,
            }
            all_results.append(result)

            # Print result
            print(f'\n{"="*60}')
            print(f'✅ {model_short_name} COMPLETE')
            print(f'{"="*60}')
            print(f'F1: {result["f1"]:.4f}')
            print(f'Precision: {result["precision"]:.4f}')
            print(f'Recall: {result["recall"]:.4f}')
            print(f'Time: {elapsed:.1f}s')

            # Check if passed
            if result['f1'] > 0.30:
                print(f'✅ PASSED smoke test (F1 > 0.30)')
            else:
                print(f'❌ FAILED smoke test (F1 = {result["f1"]:.4f})')

            # Clean up GPU memory
            del model
            del trainer
            del train_dataset
            del val_dataset
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f'\n❌ ERROR with {model_name_to_test}:')
            print(f'   {str(e)}')
            import traceback
            traceback.print_exc()

            # Record failure
            all_results.append({
                'model': model_name_to_test,
                'model_short': model_name_to_test.split('/')[-1],
                'dataset': primary_dataset,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'train_loss': 0.0,
                'eval_loss': 0.0,
                'time_seconds': 0,
                'error': str(e)
            })

            # Continue to next model
            continue

    # ============================================
    # SUMMARY OF ALL RESULTS
    # ============================================

    print('\n' + '='*60)
    print('📊 MULTI-MODEL SMOKE TEST SUMMARY')
    print('='*60)

    # Create DataFrame
    df_results = pd.DataFrame(all_results)

    # Display results
    print('\nResults:')
    print(df_results[['model_short', 'f1', 'precision', 'recall', 'time_seconds']].to_string(index=False))

    # Count passed/failed
    passed = sum(1 for r in all_results if r['f1'] > 0.30)
    failed = len(all_results) - passed

    print(f'\n{"="*60}')
    print(f'Passed: {passed}/{len(all_results)} ✅')
    print(f'Failed: {failed}/{len(all_results)} ❌')
    print(f'{"="*60}')

    # Save results
    results_file = RESULTS_DIR / f'smoke_test_all_models_{EXPERIMENT_ID}.csv'
    df_results.to_csv(results_file, index=False)
    print(f'\n💾 Results saved to: {results_file}')

    # Final verdict
    if failed == 0:
        print('\n🎉 ALL MODELS PASSED SMOKE TEST!')
        print('✅ Ready for full training')
    else:
        print(f'\n⚠️  {failed} models failed smoke test')
        print('❌ Check errors above')

    # Store results in CONFIG for later access
    CONFIG['smoke_test_results'] = all_results

else:
    print('\n⚠️  TEST_ALL_MODELS not enabled')
    print('Run Cell 4 with TEST_ALL_MODELS = True first')
