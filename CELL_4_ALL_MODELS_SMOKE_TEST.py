"""
Cell 4: Multi-Model Smoke Test
Tests all 7 models sequentially (takes ~15 minutes)
"""

# ============================================
# 🔥 MULTI-MODEL SMOKE TEST
# ============================================

import sys
import io
if sys.platform != 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print('\n' + '='*60)
print('🔥 MULTI-MODEL SMOKE TEST')
print('='*60)
print('Tests all 7 BERT models on BC2GM')
print('Time: ~15 minutes (7 models × 2 min each)')
print('='*60)

# ⭐ Set to True to test all models
TEST_ALL_MODELS = True  # Change to False for single model test

if TEST_ALL_MODELS:
    print('\n✅ TESTING ALL 7 MODELS')

    # All 7 models to test
    all_models = [
        'dmis-lab/biobert-v1.1',  # BioBERT
        'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',  # BlueBERT
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',  # PubMedBERT
        'allenai/biomed_roberta_base',  # BioMed-RoBERTa
        'emilyalsentzer/Bio_ClinicalBERT',  # Clinical-BERT
        'roberta-base',  # RoBERTa (baseline)
        'bert-base-uncased',  # BERT (baseline)
    ]

    # Store for later
    CONFIG['all_models_to_test'] = all_models
    CONFIG['current_model_index'] = 0
    CONFIG['test_results'] = []

    # Override config for smoke test
    CONFIG['max_samples_per_dataset'] = 50
    CONFIG['num_epochs'] = 1
    CONFIG['batch_size'] = 16
    CONFIG['max_length'] = 128
    CONFIG['warmup_steps'] = 10
    CONFIG['save_steps'] = 50
    CONFIG['eval_steps'] = 25
    CONFIG['use_early_stopping'] = False
    CONFIG['datasets'] = ['bc2gm']  # Just test on BC2GM

    print(f'\nModels to test: {len(all_models)}')
    for i, model in enumerate(all_models, 1):
        model_name = model.split('/')[-1]
        print(f'  {i}. {model_name}')

    print('\n⏱️  Expected time: ~15 minutes')
    print('✅ Expected: All models get F1 > 0.30')
    print('❌ If any model F1 < 0.30: That model has issues')

else:
    print('\n✅ SINGLE MODEL SMOKE TEST')
    # Single model test (original behavior)
    CONFIG['max_samples_per_dataset'] = 50
    CONFIG['num_epochs'] = 1
    CONFIG['batch_size'] = 16
    CONFIG['max_length'] = 128
    CONFIG['warmup_steps'] = 10
    CONFIG['save_steps'] = 50
    CONFIG['eval_steps'] = 25
    CONFIG['use_early_stopping'] = False

    print(f"   Model: {CONFIG['model_name']}")
    print('   Settings: 50 samples, 1 epoch')
    print('   ⏱️  Time: ~2 minutes')

print('='*60)
