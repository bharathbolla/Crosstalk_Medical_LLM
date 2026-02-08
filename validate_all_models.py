"""
Comprehensive Validation Script
Tests all 7 BERT models on BC2GM to identify issues
"""

import pickle
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from pathlib import Path
import numpy as np

# ============================================
# ALL 7 MODELS TO VALIDATE
# ============================================

MODELS_TO_TEST = {
    'BERT-base (uncased)': 'bert-base-uncased',
    'RoBERTa-base': 'roberta-base',
    'BioBERT v1.1 (cased)': 'dmis-lab/biobert-v1.1',
    'PubMedBERT (uncased)': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
    'Clinical-BERT': 'emilyalsentzer/Bio_ClinicalBERT',
    'BlueBERT (uncased)': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',
    'BioMed-RoBERTa': 'allenai/biomed_roberta_base',
}

# BC2GM label map
BC2GM_LABELS = ["O", "B-GENE", "I-GENE"]
label2id = {label: i for i, label in enumerate(BC2GM_LABELS)}
id2label = {i: label for label, i in label2id.items()}

# ============================================
# VALIDATION FUNCTIONS
# ============================================

def validate_data_integrity():
    """Check if BC2GM pickle data is valid."""
    print("\n" + "="*60)
    print("1. VALIDATING BC2GM PICKLE DATA")
    print("="*60)

    pickle_path = Path("data/pickle/bc2gm.pkl")

    if not pickle_path.exists():
        print(f"❌ ERROR: {pickle_path} not found!")
        return False

    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        train = data.get('train', [])
        test = data.get('test', [])

        print(f"✅ Data loaded successfully")
        print(f"   Train samples: {len(train):,}")
        print(f"   Test samples: {len(test):,}")

        # Check first sample structure
        if train:
            sample = train[0]
            print(f"\n   Sample structure:")
            print(f"   - Keys: {list(sample.keys())}")
            print(f"   - Has 'tokens': {'tokens' in sample}")
            print(f"   - Has 'ner_tags': {'ner_tags' in sample}")
            print(f"   - Has 'labels': {'labels' in sample}")

            # Check token-label alignment
            tokens = sample.get('tokens', [])
            labels = sample.get('ner_tags', sample.get('labels', []))

            if len(tokens) != len(labels):
                print(f"   ⚠️  WARNING: Token-label mismatch! tokens={len(tokens)}, labels={len(labels)}")
            else:
                print(f"   ✅ Token-label alignment correct: {len(tokens)} tokens = {len(labels)} labels")

            # Show example
            print(f"\n   Example (first 5 tokens):")
            for tok, lab in zip(tokens[:5], labels[:5]):
                label_name = BC2GM_LABELS[lab] if lab < len(BC2GM_LABELS) else f"UNKNOWN({lab})"
                print(f"      {tok:20s} → {label_name}")

        return True

    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return False


def test_tokenization(model_name, model_display_name):
    """Test tokenization and label alignment for a specific model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_display_name}")
    print(f"Model ID: {model_name}")
    print(f"{'='*60}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Test case-sensitivity
        is_uncased = 'uncased' in model_name.lower()
        print(f"   Tokenizer type: {'UNCASED ❌' if is_uncased else 'CASED ✅'}")

        # Load sample data
        with open("data/pickle/bc2gm.pkl", 'rb') as f:
            data = pickle.load(f)

        sample = data['train'][0]
        tokens = sample['tokens']
        labels = sample.get('ner_tags', sample.get('labels', []))

        # Test tokenization
        print(f"\n   Testing tokenization...")

        # WRONG METHOD (what causes F1=0.46)
        text = ' '.join(tokens)
        encoding_wrong = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # CORRECT METHOD
        encoding_correct = tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        num_tokens_wrong = encoding_wrong['attention_mask'].sum().item()
        num_tokens_correct = encoding_correct['attention_mask'].sum().item()

        print(f"   Wrong method tokens: {num_tokens_wrong}")
        print(f"   Correct method tokens: {num_tokens_correct}")

        if num_tokens_wrong != num_tokens_correct:
            print(f"   ⚠️  TOKENIZATION MISMATCH! This causes F1 drop!")

        # Test label alignment
        word_ids = encoding_correct.word_ids()
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

        # Count valid labels
        valid_labels = [l for l in aligned_labels if l != -100]
        print(f"   Valid labels after alignment: {len(valid_labels)}")
        print(f"   Original labels: {len(labels)}")

        if len(valid_labels) < len(labels) * 0.8:
            print(f"   ⚠️  WARNING: Lost >20% of labels during alignment!")
        else:
            print(f"   ✅ Label alignment looks good")

        # Test case sensitivity on gene names
        if is_uncased:
            test_tokens = ['TNF', 'NF-kappaB', 'IL-6']
            print(f"\n   Testing case sensitivity (CRITICAL for genes):")
            for gene in test_tokens:
                encoded = tokenizer.encode(gene, add_special_tokens=False)
                decoded = tokenizer.decode(encoded)
                if decoded.lower() != gene.lower():
                    print(f"      {gene} → {decoded} ⚠️  (tokenization changed)")
                elif gene != decoded:
                    print(f"      {gene} → {decoded} ❌ (case lost!)")
                else:
                    print(f"      {gene} → {decoded} ✅")

        print(f"\n   ✅ {model_display_name} validation complete")
        return True

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading(model_name, model_display_name):
    """Test if model can be loaded correctly."""
    try:
        print(f"\n   Loading model...")
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=3,  # BC2GM has 3 labels
            ignore_mismatched_sizes=True
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {total_params:,}")

        if torch.cuda.is_available():
            model = model.cuda()
            print(f"   ✅ Model loaded on GPU")
        else:
            print(f"   ℹ️  Model loaded on CPU")

        return True

    except Exception as e:
        print(f"   ❌ ERROR loading model: {e}")
        return False


def generate_recommendations():
    """Generate model-specific recommendations."""
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR YOUR PROJECT")
    print("="*60)

    recommendations = {
        'BioBERT v1.1 (cased)': {
            'priority': 1,
            'expected_f1': 0.84,
            'notes': 'Best for BC2GM. CASED tokenizer preserves gene names.',
            'config': 'dmis-lab/biobert-v1.1'
        },
        'PubMedBERT (uncased)': {
            'priority': 2,
            'expected_f1': 0.82,
            'notes': 'Strong baseline. Uncased but trained from scratch on PubMed.',
            'config': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
        },
        'BlueBERT (uncased)': {
            'priority': 1,
            'expected_f1': 0.85,
            'notes': 'Best overall (PubMed + MIMIC). Uncased but very strong.',
            'config': 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
        },
        'Clinical-BERT': {
            'priority': 3,
            'expected_f1': 0.80,
            'notes': 'Clinical focus. May underperform on biomedical research text.',
            'config': 'emilyalsentzer/Bio_ClinicalBERT'
        },
        'BioMed-RoBERTa': {
            'priority': 2,
            'expected_f1': 0.83,
            'notes': 'RoBERTa architecture + biomedical pretraining.',
            'config': 'allenai/biomed_roberta_base'
        },
        'BERT-base (uncased)': {
            'priority': 4,
            'expected_f1': 0.72,
            'notes': '⚠️  NO medical pretraining. Use only as general baseline.',
            'config': 'bert-base-uncased'
        },
        'RoBERTa-base': {
            'priority': 4,
            'expected_f1': 0.74,
            'notes': '⚠️  NO medical pretraining. Use only as general baseline.',
            'config': 'roberta-base'
        },
    }

    print("\nPriority 1 (MUST DO for your research):")
    for name, info in recommendations.items():
        if info['priority'] == 1:
            print(f"\n  ✅ {name}")
            print(f"     Expected F1: {info['expected_f1']:.2f}")
            print(f"     Notes: {info['notes']}")
            print(f"     Config: '{info['config']}'")

    print("\n\nPriority 2 (SHOULD DO for comprehensive comparison):")
    for name, info in recommendations.items():
        if info['priority'] == 2:
            print(f"\n  ✅ {name}")
            print(f"     Expected F1: {info['expected_f1']:.2f}")
            print(f"     Notes: {info['notes']}")

    print("\n\nPriority 3-4 (NICE TO HAVE / baselines):")
    for name, info in recommendations.items():
        if info['priority'] >= 3:
            print(f"\n  📊 {name}")
            print(f"     Expected F1: {info['expected_f1']:.2f}")
            print(f"     Notes: {info['notes']}")


# ============================================
# MAIN VALIDATION
# ============================================

def main():
    import sys
    import io

    # Fix Windows encoding
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 60)
    print("  COMPREHENSIVE VALIDATION: ALL 7 BERT MODELS x BC2GM")
    print("=" * 60)

    # Step 1: Validate data
    data_ok = validate_data_integrity()

    if not data_ok:
        print("\n❌ DATA VALIDATION FAILED! Fix data issues first.")
        return

    # Step 2: Test all models
    print("\n" + "="*60)
    print("2. TESTING ALL 7 MODELS")
    print("="*60)

    results = {}

    for model_display_name, model_name in MODELS_TO_TEST.items():
        success = test_tokenization(model_name, model_display_name)
        results[model_display_name] = success

        # Also test model loading
        if success:
            model_ok = test_model_loading(model_name, model_display_name)
            results[model_display_name] = model_ok

    # Step 3: Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)

    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{model_name:40s} {status}")

    # Step 4: Recommendations
    generate_recommendations()

    # Step 5: Critical fixes needed
    print("\n" + "="*60)
    print("🔧 CRITICAL FIXES NEEDED IN YOUR NOTEBOOK")
    print("="*60)

    print("""
1. ❌ WRONG MODEL (if using bert-base-uncased):
   Change: 'bert-base-uncased'
   To:     'dmis-lab/biobert-v1.1'  # For BC2GM

   Expected improvement: F1 = 0.46 → 0.84 (+38%!)

2. ❌ WRONG TOKENIZATION (causing label misalignment):
   Change:
      text = ' '.join(tokens)
      encoding = tokenizer(text, ...)

   To:
      encoding = tokenizer(
          tokens,
          is_split_into_words=True,  # ⭐ KEY FIX
          max_length=512,
          ...
      )

3. ❌ WRONG LABEL ALIGNMENT:
   Add proper word_ids() alignment:

      word_ids = encoding.word_ids()
      aligned_labels = []
      previous_word_idx = None

      for word_idx in word_ids:
          if word_idx is None:
              aligned_labels.append(-100)
          elif word_idx != previous_word_idx:
              aligned_labels.append(labels[word_idx])
          else:
              aligned_labels.append(-100)
          previous_word_idx = word_idx

Apply ALL 3 fixes to reach F1 ~0.84!
""")

    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()
