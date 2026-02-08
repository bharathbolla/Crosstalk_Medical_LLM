"""
Test tokenization for all model types.
Verifies is_split_into_words and word_ids() work correctly.
"""

import sys
import io
from transformers import AutoTokenizer

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def test_bert_tokenization():
    """Test BERT tokenization with is_split_into_words."""
    print("\n" + "="*60)
    print("TEST 1: BERT Tokenization")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokens = ['The', 'patient', 'has', 'diabetes']

    # Correct tokenization for NER
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )

    print(f"Input tokens: {tokens}")
    print(f"Token IDs: {encoding['input_ids'][:10]}...")
    print(f"Word IDs: {encoding.word_ids()[:10]}...")

    # Verify
    assert 'input_ids' in encoding
    assert encoding.word_ids() is not None
    assert len(encoding['input_ids']) > len(tokens)  # Has special tokens

    print("✅ BERT tokenization works with is_split_into_words=True")
    return True


def test_roberta_tokenization():
    """Test RoBERTa tokenization with add_prefix_space."""
    print("\n" + "="*60)
    print("TEST 2: RoBERTa Tokenization")
    print("="*60)

    # CRITICAL: RoBERTa needs add_prefix_space=True
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
    tokens = ['The', 'patient', 'has', 'diabetes']

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )

    print(f"Input tokens: {tokens}")
    print(f"Token IDs: {encoding['input_ids'][:10]}...")
    print(f"Word IDs: {encoding.word_ids()[:10]}...")

    assert 'input_ids' in encoding
    assert encoding.word_ids() is not None

    print("✅ RoBERTa tokenization works with add_prefix_space=True")
    return True


def test_biobert_tokenization():
    """Test BioBERT tokenization."""
    print("\n" + "="*60)
    print("TEST 3: BioBERT Tokenization")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    tokens = ['The', 'patient', 'has', 'diabetes']

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )

    print(f"Input tokens: {tokens}")
    print(f"Token IDs: {encoding['input_ids'][:10]}...")
    print(f"Word IDs: {encoding.word_ids()[:10]}...")

    assert 'input_ids' in encoding
    assert encoding.word_ids() is not None

    print("✅ BioBERT tokenization works")
    return True


def test_word_ids_alignment():
    """Test word_ids() gives correct alignment."""
    print("\n" + "="*60)
    print("TEST 4: Word IDs Alignment")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    tokens = ['diabetes', 'mellitus']  # Will be split into subwords

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )

    word_ids = encoding.word_ids()

    print(f"Input tokens: {tokens}")
    print(f"Subword tokens: {tokenizer.convert_ids_to_tokens(encoding['input_ids'])}")
    print(f"Word IDs: {word_ids}")

    # Verify word_ids structure
    assert None in word_ids  # Special tokens have None
    assert 0 in word_ids  # First word
    assert 1 in word_ids  # Second word

    print("✅ Word IDs alignment correct")
    return True


if __name__ == '__main__':
    print("="*60)
    print("TOKENIZATION TESTS")
    print("="*60)

    try:
        test_bert_tokenization()
        test_roberta_tokenization()
        test_biobert_tokenization()
        test_word_ids_alignment()

        print("\n" + "="*60)
        print("✅ ALL TOKENIZATION TESTS PASSED")
        print("="*60)
        print("\nKey takeaways:")
        print("  - Always use is_split_into_words=True for NER")
        print("  - RoBERTa needs add_prefix_space=True")
        print("  - word_ids() correctly aligns labels to subwords")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
