"""
FIXED CELL 8: Token Tracking NER Dataset
Copy this entire cell to replace your current Cell 8 in Kaggle notebook

All 3 critical fixes included:
1. is_split_into_words=True (fixes tokenization)
2. word_ids() alignment (fixes label alignment)
3. RoBERTa compatibility (add_prefix_space)
"""

from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

class TokenTrackingNERDataset(Dataset):
    """NER dataset with token counting for RQ5 - FIXED VERSION."""

    def __init__(self, data, tokenizer, max_length=512, task_name="unknown"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_name = task_name
        self.total_tokens = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Get tokens and labels
        tokens = item['tokens']  # Already split words
        labels = item.get('ner_tags', item.get('labels', [0] * len(tokens)))

        # ============================================
        # FIX 2: CORRECT TOKENIZATION
        # ============================================
        # Use is_split_into_words=True to preserve word-token alignment
        encoding = self.tokenizer(
            tokens,                      # Pass list of words directly (NOT ' '.join(tokens)!)
            is_split_into_words=True,    # ⭐ KEY FIX: Tells tokenizer these are pre-split
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # ============================================
        # FIX 3: CORRECT LABEL ALIGNMENT
        # ============================================
        # Use word_ids() to properly align labels with subword tokens

        word_ids = encoding.word_ids()  # Maps each token → original word index
        aligned_labels = []

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD])
                aligned_labels.append(-100)

            elif word_idx != previous_word_idx:
                # First subword of a word → use the original label
                if word_idx < len(labels):
                    aligned_labels.append(labels[word_idx])
                else:
                    aligned_labels.append(0)  # O tag if out of bounds

            else:
                # Continuation subword → ignore for NER (only first subword gets label)
                aligned_labels.append(-100)

            previous_word_idx = word_idx

        # Ensure correct length (pad/truncate)
        aligned_labels = aligned_labels[:self.max_length]
        while len(aligned_labels) < self.max_length:
            aligned_labels.append(-100)

        # Count tokens (for RQ5: token-controlled baseline)
        num_tokens = encoding['attention_mask'].sum().item()
        self.total_tokens += num_tokens

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels),
            'task_name': self.task_name,
            'num_tokens': num_tokens
        }


# ============================================
# TOKENIZER LOADING (handles RoBERTa)
# ============================================

# Load tokenizer with RoBERTa compatibility
print(f"\n🤖 Loading tokenizer: {CONFIG['model_name']}")

# Check if model is RoBERTa-based
is_roberta = 'roberta' in CONFIG['model_name'].lower()

if is_roberta:
    # RoBERTa requires add_prefix_space=True when using is_split_into_words
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['model_name'],
        add_prefix_space=True  # Required for RoBERTa
    )
    print(f"   ✅ RoBERTa tokenizer loaded with add_prefix_space=True")
else:
    # BERT-based models
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    print(f"   ✅ BERT-based tokenizer loaded")


# ============================================
# DATASET LOADING (same as before)
# ============================================

print("\n📦 Loading datasets...")
print("="*60)

all_train_datasets = {}
all_val_datasets = {}
all_test_datasets = {}
dataset_stats = {}

for dataset_name in CONFIG['datasets']:
    pickle_file = Path(f"data/pickle/{dataset_name}.pkl")

    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    # Apply sample limit if specified
    train_data = data['train']
    if CONFIG['max_samples_per_dataset']:
        train_data = train_data[:CONFIG['max_samples_per_dataset']]

    val_data = data.get('validation', data.get('test', train_data[:100]))
    test_data = data.get('test', val_data)

    # Create datasets with FIXED tokenization
    all_train_datasets[dataset_name] = TokenTrackingNERDataset(
        train_data, tokenizer, CONFIG['max_length'], dataset_name
    )
    all_val_datasets[dataset_name] = TokenTrackingNERDataset(
        val_data, tokenizer, CONFIG['max_length'], dataset_name
    )
    all_test_datasets[dataset_name] = TokenTrackingNERDataset(
        test_data, tokenizer, CONFIG['max_length'], dataset_name
    )

    # Calculate unique label count
    all_labels = set()
    for item in train_data:
        all_labels.update(item.get('ner_tags', item.get('labels', [])))
    num_labels = len(all_labels)

    dataset_stats[dataset_name] = {
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'num_labels': num_labels,
    }

    print(f"\n{dataset_name.upper()}:")
    print(f"  Train: {len(train_data):,} samples")
    print(f"  Val: {len(val_data):,} samples")
    print(f"  Test: {len(test_data):,} samples")
    print(f"  Labels: {num_labels}")

print("\n" + "="*60)
print(f"✅ Loaded {len(CONFIG['datasets'])} dataset(s) with FIXED tokenization")

# Save dataset stats
stats_path = RESULTS_DIR / f"dataset_stats_{EXPERIMENT_ID}.json"
with open(stats_path, 'w') as f:
    json.dump(dataset_stats, f, indent=2)
