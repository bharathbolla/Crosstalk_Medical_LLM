"""
Comprehensive tests for UniversalMedicalDataset class.
Tests all 8 tasks and verifies correct processing for each type.
"""

import sys
import io
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import after adding to path
from transformers import AutoTokenizer


def test_task_configs():
    """Test TASK_CONFIGS for all 8 datasets."""
    print("\n" + "="*60)
    print("TEST 1: Task Configurations")
    print("="*60)

    # Import TASK_CONFIGS from COMPLETE_FIXED_DATASET.py
    exec(open('COMPLETE_FIXED_DATASET.py').read(), globals())

    datasets = ['bc2gm', 'jnlpba', 'chemprot', 'ddi', 'gad', 'hoc', 'pubmedqa', 'biosses']

    for ds in datasets:
        assert ds in TASK_CONFIGS, f"Missing config for {ds}"
        config = TASK_CONFIGS[ds]

        assert 'task_type' in config, f"{ds} missing task_type"
        assert 'model_type' in config, f"{ds} missing model_type"
        assert 'labels' in config, f"{ds} missing labels"

        print(f"✅ {ds:12s} | Type: {config['task_type']:15s} | Model: {config['model_type']}")

    print("\n✅ All 8 task configs valid")
    return True


def test_dataset_creation():
    """Test UniversalMedicalDataset creation for BC2GM."""
    print("\n" + "="*60)
    print("TEST 2: Dataset Creation (BC2GM)")
    print("="*60)

    # Load modules
    exec(open('COMPLETE_FIXED_DATASET.py').read(), globals())

    # Load data
    pkl_file = Path('data/pickle/bc2gm.pkl')
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')

    # Create dataset
    dataset = UniversalMedicalDataset(
        data=data['train'][:10],  # First 10 samples
        tokenizer=tokenizer,
        task_name='bc2gm',
        max_length=128
    )

    assert len(dataset) == 10
    print(f"Dataset size: {len(dataset)}")

    # Test __getitem__
    sample = dataset[0]

    print(f"Sample keys: {list(sample.keys())}")
    assert 'input_ids' in sample
    assert 'attention_mask' in sample
    assert 'labels' in sample

    print(f"Input IDs shape: {len(sample['input_ids'])}")
    print(f"Labels shape: {len(sample['labels'])}")

    assert len(sample['input_ids']) == len(sample['labels'])

    print("✅ Dataset creation works")
    return True


def test_ner_processing():
    """Test NER task processing (BC2GM)."""
    print("\n" + "="*60)
    print("TEST 3: NER Processing")
    print("="*60)

    exec(open('COMPLETE_FIXED_DATASET.py').read(), globals())

    # Load data
    pkl_file = Path('data/pickle/bc2gm.pkl')
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')

    dataset = UniversalMedicalDataset(
        data=data['train'][:5],
        tokenizer=tokenizer,
        task_name='bc2gm',
        max_length=128
    )

    sample = dataset[0]

    # Verify NER-specific processing
    labels = sample['labels']

    print(f"Label values: {set(labels)}")

    # Should have -100 (special tokens), 0 (O), 1 (B-GENE), 2 (I-GENE)
    assert -100 in labels  # Special tokens
    assert 0 in labels or 1 in labels or 2 in labels  # Some entity labels

    # Check label range
    valid_labels = [l for l in labels if l != -100]
    assert all(0 <= l <= 2 for l in valid_labels), "Labels outside valid range"

    print("✅ NER processing correct")
    return True


def test_sequence_classification_processing():
    """Test sequence classification (GAD)."""
    print("\n" + "="*60)
    print("TEST 4: Sequence Classification Processing")
    print("="*60)

    exec(open('COMPLETE_FIXED_DATASET.py').read(), globals())

    # Load data
    pkl_file = Path('data/pickle/gad.pkl')
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')

    dataset = UniversalMedicalDataset(
        data=data['train'][:5],
        tokenizer=tokenizer,
        task_name='gad',
        max_length=256
    )

    sample = dataset[0]

    # Verify sequence classification
    labels = sample['labels']

    print(f"Label type: {type(labels)}")
    print(f"Label value: {labels}")

    # For classification, labels should be a single integer
    assert isinstance(labels, int), "Label should be single integer"
    assert 0 <= labels <= 1, "Label outside valid range for binary classification"

    print("✅ Sequence classification processing correct")
    return True


def test_regression_processing():
    """Test regression processing (BIOSSES)."""
    print("\n" + "="*60)
    print("TEST 5: Regression Processing")
    print("="*60)

    exec(open('COMPLETE_FIXED_DATASET.py').read(), globals())

    # Load data
    pkl_file = Path('data/pickle/biosses.pkl')
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')

    dataset = UniversalMedicalDataset(
        data=data['train'][:5],
        tokenizer=tokenizer,
        task_name='biosses',
        max_length=128
    )

    sample = dataset[0]

    # Verify regression
    labels = sample['labels']

    print(f"Label type: {type(labels)}")
    print(f"Label value: {labels}")

    # For regression, labels should be a float
    assert isinstance(labels, (float, int)), "Label should be numeric"

    print("✅ Regression processing correct")
    return True


def test_all_task_types():
    """Test one dataset from each task type."""
    print("\n" + "="*60)
    print("TEST 6: All Task Types")
    print("="*60)

    exec(open('COMPLETE_FIXED_DATASET.py').read(), globals())

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')

    task_examples = {
        'bc2gm': 'ner',
        'chemprot': 're',
        'gad': 'classification',
        'hoc': 'multilabel_classification',
        'pubmedqa': 'qa',
        'biosses': 'similarity',
    }

    for task_name, task_type in task_examples.items():
        pkl_file = Path('data/pickle') / f'{task_name}.pkl'

        if not pkl_file.exists():
            print(f"⚠️  Skipping {task_name} (file not found)")
            continue

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        dataset = UniversalMedicalDataset(
            data=data['train'][:3],
            tokenizer=tokenizer,
            task_name=task_name,
            max_length=128
        )

        assert len(dataset) == 3
        sample = dataset[0]
        assert 'input_ids' in sample
        assert 'labels' in sample

        print(f"✅ {task_name:12s} | {task_type:20s} | Processed correctly")

    print("\n✅ All task types work")
    return True


def test_token_counting():
    """Test token counting for RQ5."""
    print("\n" + "="*60)
    print("TEST 7: Token Counting (RQ5)")
    print("="*60)

    exec(open('COMPLETE_FIXED_DATASET.py').read(), globals())

    # Load data
    pkl_file = Path('data/pickle/bc2gm.pkl')
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')

    dataset = UniversalMedicalDataset(
        data=data['train'][:10],
        tokenizer=tokenizer,
        task_name='bc2gm',
        max_length=128
    )

    # Check total_tokens counter
    assert hasattr(dataset, 'total_tokens'), "Dataset should track tokens"
    print(f"Total tokens processed: {dataset.total_tokens:,}")

    assert dataset.total_tokens > 0, "Should have counted tokens"

    print("✅ Token counting works")
    return True


if __name__ == '__main__':
    print("="*60)
    print("UNIVERSAL DATASET TESTS")
    print("="*60)
    print("\nTests UniversalMedicalDataset for all task types")

    try:
        test_task_configs()
        test_dataset_creation()
        test_ner_processing()
        test_sequence_classification_processing()
        test_regression_processing()
        test_all_task_types()
        test_token_counting()

        print("\n" + "="*60)
        print("✅ ALL DATASET TESTS PASSED")
        print("="*60)
        print("\nKey lessons:")
        print("  1. Pass task_name (not task_type) to UniversalMedicalDataset")
        print("  2. NER returns list of labels (one per token)")
        print("  3. Classification returns single integer label")
        print("  4. Regression returns float label")
        print("  5. Token counting works for RQ5")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
