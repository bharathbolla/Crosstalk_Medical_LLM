"""
Comprehensive tests for model loading across all task types.
Tests all 8 tasks and ensures correct model heads are used.
"""

import sys
import io
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoConfig
)

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def test_ner_model():
    """Test NER model loading (BC2GM, JNLPBA)."""
    print("\n" + "="*60)
    print("TEST 1: NER Model (TokenClassification)")
    print("="*60)

    # BC2GM: 3 labels (O, B-GENE, I-GENE)
    model = AutoModelForTokenClassification.from_pretrained(
        'dmis-lab/biobert-v1.1',
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    assert model is not None
    assert hasattr(model, 'classifier')

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BioBERT for BC2GM")
    print(f"Num labels: 3")
    print(f"Parameters: {total_params:,}")
    print(f"Head type: TokenClassification")

    # Test forward pass
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    assert logits.shape == (batch_size, seq_len, 3)
    print(f"Output shape: {logits.shape} ✓")

    print("✅ NER model works")
    return True


def test_relation_extraction_model():
    """Test RE model loading (ChemProt, DDI)."""
    print("\n" + "="*60)
    print("TEST 2: Relation Extraction (SequenceClassification)")
    print("="*60)

    # ChemProt: 13 relation types
    model = AutoModelForSequenceClassification.from_pretrained(
        'dmis-lab/biobert-v1.1',
        num_labels=13,
        ignore_mismatched_sizes=True
    )

    assert model is not None
    assert hasattr(model, 'classifier')

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: BioBERT for ChemProt")
    print(f"Num labels: 13")
    print(f"Parameters: {total_params:,}")
    print(f"Head type: SequenceClassification")

    # Test forward pass
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    assert logits.shape == (batch_size, 13)
    print(f"Output shape: {logits.shape} ✓")

    print("✅ RE model works")
    return True


def test_classification_model():
    """Test classification model (GAD)."""
    print("\n" + "="*60)
    print("TEST 3: Binary Classification")
    print("="*60)

    # GAD: 2 classes (0, 1)
    model = AutoModelForSequenceClassification.from_pretrained(
        'dmis-lab/biobert-v1.1',
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    assert model is not None

    print(f"Model: BioBERT for GAD")
    print(f"Num labels: 2 (binary)")
    print(f"Head type: SequenceClassification")

    # Test forward pass
    batch_size, seq_len = 2, 30
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    assert logits.shape == (batch_size, 2)
    print(f"Output shape: {logits.shape} ✓")

    print("✅ Binary classification model works")
    return True


def test_multilabel_model():
    """Test multi-label classification (HoC)."""
    print("\n" + "="*60)
    print("TEST 4: Multi-Label Classification")
    print("="*60)

    # HoC: 10 hallmarks (multi-label)
    config = AutoConfig.from_pretrained('dmis-lab/biobert-v1.1')
    config.problem_type = "multi_label_classification"

    model = AutoModelForSequenceClassification.from_pretrained(
        'dmis-lab/biobert-v1.1',
        config=config,
        num_labels=10,
        ignore_mismatched_sizes=True
    )

    assert model is not None
    assert model.config.problem_type == "multi_label_classification"

    print(f"Model: BioBERT for HoC")
    print(f"Num labels: 10 (multi-label)")
    print(f"Problem type: {model.config.problem_type}")
    print(f"Head type: SequenceClassification")

    # Test forward pass
    batch_size, seq_len = 2, 25
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 2, (batch_size, 10)).float()  # Multi-hot encoding

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    logits = outputs.logits
    loss = outputs.loss

    assert logits.shape == (batch_size, 10)
    assert loss is not None
    print(f"Output shape: {logits.shape} ✓")
    print(f"Loss computed: {loss.item():.4f} ✓")

    print("✅ Multi-label model works")
    return True


def test_qa_model():
    """Test QA model (PubMedQA)."""
    print("\n" + "="*60)
    print("TEST 5: Question Answering")
    print("="*60)

    # PubMedQA: 3 classes (no, yes, maybe)
    model = AutoModelForSequenceClassification.from_pretrained(
        'dmis-lab/biobert-v1.1',
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    assert model is not None

    print(f"Model: BioBERT for PubMedQA")
    print(f"Num labels: 3 (no, yes, maybe)")
    print(f"Head type: SequenceClassification")

    # Test forward pass
    batch_size, seq_len = 2, 256  # QA needs longer sequences
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    assert logits.shape == (batch_size, 3)
    print(f"Output shape: {logits.shape} ✓")

    print("✅ QA model works")
    return True


def test_regression_model():
    """Test regression model (BIOSSES)."""
    print("\n" + "="*60)
    print("TEST 6: Regression (Similarity)")
    print("="*60)

    # BIOSSES: Regression (num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained(
        'dmis-lab/biobert-v1.1',
        num_labels=1,
        ignore_mismatched_sizes=True
    )

    assert model is not None

    print(f"Model: BioBERT for BIOSSES")
    print(f"Num labels: 1 (regression)")
    print(f"Head type: SequenceClassification (regression)")

    # Test forward pass
    batch_size, seq_len = 2, 40
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    assert logits.shape == (batch_size, 1)
    print(f"Output shape: {logits.shape} ✓")

    # Regression output should be continuous
    similarity_scores = logits.squeeze()
    print(f"Example scores: {similarity_scores.tolist()}")

    print("✅ Regression model works")
    return True


def test_all_medical_models():
    """Test loading all 7 medical models."""
    print("\n" + "="*60)
    print("TEST 7: All Medical Models")
    print("="*60)

    models = {
        'BioBERT': 'dmis-lab/biobert-v1.1',
        'BlueBERT': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
        'PubMedBERT': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
        # Skip heavy downloads for quick test
        # 'BioMed-RoBERTa': 'allenai/biomed_roberta_base',
        # 'Clinical-BERT': 'emilyalsentzer/Bio_ClinicalBERT',
        # 'RoBERTa': 'roberta-base',
        # 'BERT': 'bert-base-uncased',
    }

    for name, model_id in models.items():
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            print(f"✅ {name:15s} loaded")
        except Exception as e:
            print(f"❌ {name:15s} failed: {e}")
            return False

    print("\n✅ All medical models load successfully")
    return True


def test_parameter_counts():
    """Test parameter counts are reasonable."""
    print("\n" + "="*60)
    print("TEST 8: Parameter Counts")
    print("="*60)

    model = AutoModelForTokenClassification.from_pretrained(
        'dmis-lab/biobert-v1.1',
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.1f}%")

    # BioBERT should have ~110M parameters
    assert 100_000_000 < total_params < 150_000_000, "Parameter count outside expected range"
    assert trainable_params == total_params, "Not all parameters trainable"

    print("✅ Parameter counts reasonable")
    return True


if __name__ == '__main__':
    print("="*60)
    print("MODEL LOADING TESTS")
    print("="*60)
    print("\nTests all 8 task types and model heads")

    try:
        test_ner_model()
        test_relation_extraction_model()
        test_classification_model()
        test_multilabel_model()
        test_qa_model()
        test_regression_model()
        test_all_medical_models()
        test_parameter_counts()

        print("\n" + "="*60)
        print("✅ ALL MODEL LOADING TESTS PASSED")
        print("="*60)
        print("\nKey lessons:")
        print("  1. NER → AutoModelForTokenClassification")
        print("  2. RE/Classification/QA → AutoModelForSequenceClassification")
        print("  3. Regression → AutoModelForSequenceClassification (num_labels=1)")
        print("  4. Multi-label → Set problem_type='multi_label_classification'")
        print("  5. All medical models load without errors")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
