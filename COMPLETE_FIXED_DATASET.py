"""
COMPLETE FIXED DATASET CLASS
Handles ALL 8 tasks: NER, RE, Classification, QA, Similarity

This replaces Cell 8 in your Kaggle notebook
Works with all 7 BERT models and all 8 datasets
"""

from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch
from pathlib import Path
import pickle

# ============================================
# TASK CONFIGURATIONS
# ============================================

TASK_CONFIGS = {
    # NER tasks - token classification
    'bc2gm': {
        'task_type': 'ner',
        'labels': ['O', 'B-GENE', 'I-GENE'],
        'model_type': 'token_classification',
    },
    'jnlpba': {
        'task_type': 'ner',
        'labels': [
            'O',
            'B-DNA', 'I-DNA',
            'B-RNA', 'I-RNA',
            'B-cell_line', 'I-cell_line',
            'B-cell_type', 'I-cell_type',
            'B-protein', 'I-protein'
        ],
        'model_type': 'token_classification',
    },

    # Relation Extraction - sequence classification
    'chemprot': {
        'task_type': 're',
        'labels': ['Agonist', 'Antagonist', 'Cofactor', 'Downregulator', 'Modulator',
                    'Not', 'Part_of', 'Regulator', 'Substrate', 'Undefined', 'Upregulator'],
        'model_type': 'sequence_classification',
    },
    'ddi': {
        'task_type': 're',
        'labels': ['ADVISE', 'EFFECT', 'INT', 'MECHANISM'],
        'model_type': 'sequence_classification',
    },

    # Classification tasks
    'gad': {
        'task_type': 'classification',
        'labels': [0, 1],  # Binary: 0=no association, 1=association
        'model_type': 'sequence_classification',
    },
    'hoc': {
        'task_type': 'multilabel_classification',
        'labels': ['activating invasion and metastasis', 'avoiding immune destruction',
                    'cellular energetics', 'enabling replicative immortality',
                    'evading growth suppressors', 'genomic instability and mutation',
                    'inducing angiogenesis', 'resisting cell death',
                    'sustaining proliferative signaling', 'tumor promoting inflammation'],
        'model_type': 'sequence_classification',
        'problem_type': 'multi_label_classification',
    },

    # QA task
    'pubmedqa': {
        'task_type': 'qa',
        'labels': ['maybe', 'no', 'yes'],
        'model_type': 'sequence_classification',
    },

    # Similarity/Regression task
    'biosses': {
        'task_type': 'similarity',
        'labels': None,  # Regression - no labels, just scores
        'model_type': 'regression',
    },
}


# ============================================
# UNIVERSAL DATASET CLASS
# ============================================

class UniversalMedicalDataset(Dataset):
    """
    Universal dataset that handles ALL 8 task types.
    Auto-detects task type and applies appropriate processing.
    """

    def __init__(self, data, tokenizer, task_name, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.max_length = max_length
        self.total_tokens = 0

        # Get task configuration
        self.task_config = TASK_CONFIGS.get(task_name, {})
        self.task_type = self.task_config.get('task_type', 'unknown')
        self.labels = self.task_config.get('labels', [])

        # Create label mappings
        if self.labels:
            self.label2id = {label: i for i, label in enumerate(self.labels)}
            self.id2label = {i: label for label, i in self.label2id.items()}
        else:
            self.label2id = {}
            self.id2label = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Route to appropriate processing based on task type
        if self.task_type == 'ner':
            return self._process_ner(item)
        elif self.task_type == 're':
            return self._process_relation_extraction(item)
        elif self.task_type in ['classification', 'multilabel_classification']:
            return self._process_classification(item)
        elif self.task_type == 'qa':
            return self._process_qa(item)
        elif self.task_type == 'similarity':
            return self._process_similarity(item)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    # ============================================
    # NER PROCESSING (BC2GM, JNLPBA)
    # ============================================

    def _process_ner(self, item):
        """Process NER tasks with token-level labels."""

        tokens = item['tokens']
        labels = item.get('ner_tags', item.get('labels', [0] * len(tokens)))

        # Tokenize with word alignment
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Align labels with subword tokens
        word_ids = encoding.word_ids()
        aligned_labels = []

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(labels):
                    aligned_labels.append(labels[word_idx])
                else:
                    aligned_labels.append(0)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        # Ensure correct length
        aligned_labels = aligned_labels[:self.max_length]
        while len(aligned_labels) < self.max_length:
            aligned_labels.append(-100)

        # Count tokens
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
    # RELATION EXTRACTION (ChemProt, DDI)
    # ============================================

    def _process_relation_extraction(self, item):
        """Process RE tasks with sentence-level labels."""

        # Get text and relation label
        if 'tokens' in item:
            tokens = item['tokens']
            text = ' '.join(tokens)
        else:
            text = item.get('text', item.get('sentence', ''))

        # Add entity markers so model knows which pair to classify
        entity1 = item.get('entity1', '')
        entity2 = item.get('entity2', '')
        if entity1 and entity2:
            text = f"[E1] {entity1} [/E1] [E2] {entity2} [/E2] {text}"

        # Get relation label
        relation = item.get('relation', item.get('label', 0))

        # Convert string label to ID if needed
        if isinstance(relation, str) and relation in self.label2id:
            relation = self.label2id[relation]
        elif isinstance(relation, str):
            relation = 0  # Default to first class if unknown

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Count tokens
        num_tokens = encoding['attention_mask'].sum().item()
        self.total_tokens += num_tokens

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(relation),
            'task_name': self.task_name,
            'num_tokens': num_tokens
        }

    # ============================================
    # CLASSIFICATION (GAD, HoC)
    # ============================================

    def _process_classification(self, item):
        """Process classification tasks."""

        # Get text
        if 'tokens' in item:
            text = ' '.join(item['tokens'])
        else:
            text = item.get('text', item.get('sentence', ''))

        # Get label(s)
        label = item.get('label', item.get('labels', 0))

        # For multi-label (HoC), convert to binary vector
        if self.task_type == 'multilabel_classification':
            if isinstance(label, list) and len(label) == len(self.labels):
                # Already a multi-hot vector from pickle
                label = torch.tensor([float(x) for x in label], dtype=torch.float)
            elif isinstance(label, list):
                # List of label IDs - convert to multi-hot
                label_vector = [0.0] * len(self.labels)
                for l in label:
                    if isinstance(l, int) and l < len(self.labels):
                        label_vector[l] = 1.0
                label = torch.tensor(label_vector, dtype=torch.float)
            else:
                # Single label - convert to one-hot
                label_vector = [0.0] * len(self.labels)
                if isinstance(label, int) and label < len(self.labels):
                    label_vector[label] = 1.0
                label = torch.tensor(label_vector, dtype=torch.float)
        else:
            # Single-label classification
            if isinstance(label, str) and label in self.label2id:
                label = self.label2id[label]
            label = torch.tensor(label)

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Count tokens
        num_tokens = encoding['attention_mask'].sum().item()
        self.total_tokens += num_tokens

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label,
            'task_name': self.task_name,
            'num_tokens': num_tokens
        }

    # ============================================
    # QA (PubMedQA)
    # ============================================

    def _process_qa(self, item):
        """Process QA tasks with question-context pairs."""

        # Get text - pickle stores pre-joined text, or separate question/context
        text = item.get('text', '')
        if not text:
            question = item.get('question', '')
            context = item.get('context', '')
            if isinstance(context, list):
                context = ' '.join(context)
            text = f"{question} {context}".strip()

        # Get answer label
        answer = item.get('label', item.get('answer', item.get('final_decision', 'maybe')))

        # Convert answer to ID
        if isinstance(answer, str):
            answer = answer.lower()
            if answer in self.label2id:
                answer = self.label2id[answer]
            else:
                answer = 2  # Default to 'maybe' if unknown

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Count tokens
        num_tokens = encoding['attention_mask'].sum().item()
        self.total_tokens += num_tokens

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(answer),
            'task_name': self.task_name,
            'num_tokens': num_tokens
        }

    # ============================================
    # SIMILARITY (BIOSSES)
    # ============================================

    def _process_similarity(self, item):
        """Process similarity tasks with sentence pairs."""

        # Get sentence pair
        sentence1 = item.get('sentence1', item.get('sentence_1', ''))
        sentence2 = item.get('sentence2', item.get('sentence_2', ''))

        # Get similarity score (0-4 range typically)
        score = item.get('score', item.get('similarity', 0.0))

        # Ensure score is float
        score = float(score)

        # Tokenize sentence pair
        encoding = self.tokenizer(
            sentence1,
            sentence2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Count tokens
        num_tokens = encoding['attention_mask'].sum().item()
        self.total_tokens += num_tokens

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(score, dtype=torch.float),
            'task_name': self.task_name,
            'num_tokens': num_tokens
        }


# ============================================
# TOKENIZER LOADING (with RoBERTa support)
# ============================================

def load_tokenizer(model_name):
    """Load tokenizer with RoBERTa compatibility."""

    is_roberta = 'roberta' in model_name.lower()

    if is_roberta:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            add_prefix_space=True  # Required for RoBERTa with is_split_into_words
        )
        print(f"   ✅ RoBERTa tokenizer loaded with add_prefix_space=True")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"   ✅ BERT-based tokenizer loaded")

    return tokenizer


# ============================================
# DATASET LOADING
# ============================================

def load_all_datasets(dataset_names, tokenizer, max_length=512, max_samples=None):
    """
    Load all datasets with automatic task type detection.

    Returns:
        train_datasets: dict of train datasets
        val_datasets: dict of validation datasets
        test_datasets: dict of test datasets
        dataset_stats: dict of dataset statistics
    """

    train_datasets = {}
    val_datasets = {}
    test_datasets = {}
    dataset_stats = {}

    for dataset_name in dataset_names:
        print(f"\n📦 Loading {dataset_name}...")

        # Load pickle data
        pickle_file = Path(f"data/pickle/{dataset_name}.pkl")

        if not pickle_file.exists():
            print(f"   ⚠️  Pickle file not found: {pickle_file}")
            continue

        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)

        # Get splits
        train_data = data.get('train', [])
        val_data = data.get('validation', data.get('val', data.get('dev', [])))
        test_data = data.get('test', [])

        # If no val, use part of train
        if not val_data and train_data:
            val_data = train_data[:min(100, len(train_data) // 10)]

        # If no test, use val
        if not test_data:
            test_data = val_data

        # Apply sample limit
        if max_samples:
            train_data = train_data[:max_samples]

        # Create datasets
        train_datasets[dataset_name] = UniversalMedicalDataset(
            train_data, tokenizer, dataset_name, max_length
        )
        val_datasets[dataset_name] = UniversalMedicalDataset(
            val_data, tokenizer, dataset_name, max_length
        )
        test_datasets[dataset_name] = UniversalMedicalDataset(
            test_data, tokenizer, dataset_name, max_length
        )

        # Get task info
        task_config = TASK_CONFIGS.get(dataset_name, {})
        task_type = task_config.get('task_type', 'unknown')
        num_labels = len(task_config.get('labels', [])) if task_config.get('labels') else 1

        dataset_stats[dataset_name] = {
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'task_type': task_type,
            'num_labels': num_labels,
            'model_type': task_config.get('model_type', 'unknown'),
        }

        print(f"   ✅ {dataset_name}")
        print(f"      Task type: {task_type}")
        print(f"      Train: {len(train_data):,} samples")
        print(f"      Val: {len(val_data):,} samples")
        print(f"      Test: {len(test_data):,} samples")
        print(f"      Labels: {num_labels}")

    return train_datasets, val_datasets, test_datasets, dataset_stats


# ============================================
# USAGE EXAMPLE (for Cell 8 in notebook)
# ============================================

if __name__ == "__main__":
    """
    Example usage - copy this to Cell 8 in your Kaggle notebook
    """

    print("\n🤖 Loading tokenizer...")
    tokenizer = load_tokenizer(CONFIG['model_name'])

    print("\n📦 Loading datasets...")
    print("="*60)

    train_datasets, val_datasets, test_datasets, dataset_stats = load_all_datasets(
        dataset_names=CONFIG['datasets'],
        tokenizer=tokenizer,
        max_length=CONFIG.get('max_length', 512),
        max_samples=CONFIG.get('max_samples_per_dataset', None)
    )

    print("\n" + "="*60)
    print(f"✅ Loaded {len(CONFIG['datasets'])} dataset(s)")
    print("="*60)

    # Show dataset stats
    for dataset_name, stats in dataset_stats.items():
        print(f"\n{dataset_name.upper()}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
