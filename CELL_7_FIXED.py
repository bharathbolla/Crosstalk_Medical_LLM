"""
Fixed Cell 7: Load Datasets
Correct pickle file path: data/pickle/{dataset}.pkl
"""

# ============================================
# CELL 7: LOAD DATASETS (FIXED)
# ============================================

import pickle
from pathlib import Path

print('\n📊 Loading datasets...')

primary_dataset = CONFIG['datasets'][0]
max_samples = CONFIG['max_samples_per_dataset']

# FIXED: Correct pickle file path
pickle_file = Path('data/pickle') / f'{primary_dataset}.pkl'

print(f'   Loading from: {pickle_file}')

# Load pickle file
with open(pickle_file, 'rb') as f:
    raw_data = pickle.load(f)

# Limit samples if smoke test
if max_samples:
    raw_data['train'] = raw_data['train'][:max_samples]
    raw_data['validation'] = raw_data['validation'][:max_samples//5]

print(f'   Dataset: {primary_dataset}')
print(f"   Train samples: {len(raw_data['train']):,}")
print(f"   Validation samples: {len(raw_data['validation']):,}")

# Create datasets using UniversalMedicalDataset
task_config = TASK_CONFIGS[primary_dataset]

train_dataset = UniversalMedicalDataset(
    data=raw_data['train'],
    tokenizer=tokenizer,
    task_type=task_config['task_type'],
    labels=task_config['labels'],
    max_length=CONFIG['max_length']
)

val_dataset = UniversalMedicalDataset(
    data=raw_data['validation'],
    tokenizer=tokenizer,
    task_type=task_config['task_type'],
    labels=task_config['labels'],
    max_length=CONFIG['max_length']
)

# Store dataset stats
dataset_stats = {
    primary_dataset: {
        'task_type': task_config['task_type'],
        'model_type': task_config['model_type'],
        'num_labels': len(task_config['labels']) if task_config['labels'] else 1,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
    }
}

print(f"   ✅ Created UniversalMedicalDataset")
print(f"   Task type: {task_config['task_type']}")
print(f"   Num labels: {dataset_stats[primary_dataset]['num_labels']}")
print('='*60)
