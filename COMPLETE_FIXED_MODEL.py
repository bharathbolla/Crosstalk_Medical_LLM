"""
COMPLETE FIXED MODEL LOADING
Automatically selects correct model head for each task type

This replaces Cell 9 (model loading) in your Kaggle notebook
Works with all 7 BERT models and all 8 task types
"""

from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoConfig
)
import torch

# ============================================
# MODEL LOADING FUNCTIONS
# ============================================

def load_model_for_task(model_name, task_name, dataset_stats):
    """
    Load appropriate model based on task type.

    Args:
        model_name: HuggingFace model ID
        task_name: Dataset name (e.g., 'bc2gm')
        dataset_stats: Statistics dict from dataset loading

    Returns:
        model: Loaded model with appropriate head
    """

    # Get task info
    task_info = dataset_stats[task_name]
    model_type = task_info['model_type']
    num_labels = task_info['num_labels']
    task_type = task_info['task_type']

    print(f"\n🤖 Loading model for {task_name}...")
    print(f"   Model: {model_name}")
    print(f"   Task type: {task_type}")
    print(f"   Model head: {model_type}")
    print(f"   Number of labels: {num_labels}")

    # Load model with appropriate head
    if model_type == 'token_classification':
        # NER tasks (BC2GM, JNLPBA)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    elif model_type == 'sequence_classification':
        # RE, Classification, QA tasks
        config = AutoConfig.from_pretrained(model_name)

        # For multi-label classification (HoC)
        if task_type == 'multilabel_classification':
            config.problem_type = "multi_label_classification"

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    elif model_type == 'regression':
        # Similarity tasks (BIOSSES)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,  # Regression: single output
            ignore_mismatched_sizes=True
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   ✅ Model loaded")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")

    return model


def load_model_for_multitask(model_name, dataset_stats, datasets):
    """
    Load model for multi-task learning.

    For multi-task, we need to handle multiple task types.
    Strategy: Use the primary task's head, or create task-specific heads.

    Args:
        model_name: HuggingFace model ID
        dataset_stats: Dict of dataset statistics
        datasets: List of dataset names

    Returns:
        model: Multi-task model (for now, just use primary task)
    """

    # For simplicity, use the first dataset's task type
    # TODO: Implement proper multi-task architecture with task-specific heads
    primary_task = datasets[0]

    print(f"\n🤖 Loading model for multi-task learning...")
    print(f"   Model: {model_name}")
    print(f"   Primary task: {primary_task}")
    print(f"   All tasks: {datasets}")

    # Load model for primary task
    model = load_model_for_task(model_name, primary_task, dataset_stats)

    print(f"   ⚠️  Note: Using {primary_task} head for all tasks")
    print(f"   ⚠️  TODO: Implement task-specific heads for proper MTL")

    return model


# ============================================
# USAGE EXAMPLE (for Cell 9 in notebook)
# ============================================

def example_single_task():
    """Example: Load model for single-task learning."""

    # Assume these are available from previous cells
    model_name = CONFIG['model_name']
    primary_dataset = CONFIG['datasets'][0]

    # Load model with automatic task detection
    model = load_model_for_task(model_name, primary_dataset, dataset_stats)

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"\n   ✅ Model moved to GPU")
    else:
        print(f"\n   ℹ️  Model on CPU")

    return model


def example_multitask():
    """Example: Load model for multi-task learning."""

    model_name = CONFIG['model_name']
    datasets = CONFIG['datasets']

    # Load multi-task model
    model = load_model_for_multitask(model_name, dataset_stats, datasets)

    # Move to GPU
    if torch.cuda.is_available():
        model = model.cuda()

    return model


# ============================================
# COMPLETE CELL 9 CODE
# ============================================

"""
Copy this to Cell 9 in your Kaggle notebook:
"""

CELL_9_CODE = '''
# ============================================
# CELL 9: Load Model with Automatic Task Detection
# ============================================

from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoConfig
)

def load_model_for_task(model_name, task_name, dataset_stats):
    """Load appropriate model based on task type."""

    task_info = dataset_stats[task_name]
    model_type = task_info['model_type']
    num_labels = task_info['num_labels']
    task_type = task_info['task_type']

    print(f"\\n🤖 Loading model for {task_name}...")
    print(f"   Model: {model_name}")
    print(f"   Task type: {task_type}")
    print(f"   Number of labels: {num_labels}")

    # Select appropriate model head
    if model_type == 'token_classification':
        # NER tasks
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    elif model_type == 'sequence_classification':
        # RE, Classification, QA
        config = AutoConfig.from_pretrained(model_name)

        if task_type == 'multilabel_classification':
            config.problem_type = "multi_label_classification"

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    elif model_type == 'regression':
        # Similarity
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   ✅ Model loaded")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model


# Load model for primary task
primary_dataset = CONFIG['datasets'][0]
model = load_model_for_task(CONFIG['model_name'], primary_dataset, dataset_stats)

# Move to GPU
if torch.cuda.is_available():
    model = model.cuda()
    print(f"\\n✅ Model moved to GPU")

print("="*60)
'''

if __name__ == "__main__":
    print("="*60)
    print("MODEL LOADING CODE FOR ALL 8 TASKS")
    print("="*60)
    print("\nThis code automatically:")
    print("  1. Detects task type (NER, RE, Classification, QA, Similarity)")
    print("  2. Loads appropriate model head:")
    print("     - TokenClassification for NER")
    print("     - SequenceClassification for RE/Classification/QA")
    print("     - Regression for Similarity")
    print("  3. Configures multi-label for HoC")
    print("  4. Works with all 7 BERT models")
    print("\nCopy CELL_9_CODE to your notebook Cell 9!")
    print("="*60)
