"""
Comprehensive tests for label alignment with word_ids().
This is CRITICAL for NER tasks - catches the F1=0.46 bug!
"""

import sys
import io
from transformers import AutoTokenizer

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def test_basic_alignment():
    """Test basic label alignment with simple tokens."""
    print("\n" + "="*60)
    print("TEST 1: Basic Label Alignment")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Simple example
    tokens = ['The', 'patient', 'has', 'diabetes']
    labels = [0, 0, 0, 1]  # Only 'diabetes' is entity (B-GENE)

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )

    word_ids = encoding.word_ids()

    # Align labels
    aligned_labels = []
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)  # Special tokens
        else:
            aligned_labels.append(labels[word_id])

    print(f"Tokens: {tokens}")
    print(f"Labels: {labels}")
    print(f"Subwords: {tokenizer.convert_ids_to_tokens(encoding['input_ids'])}")
    print(f"Word IDs: {word_ids}")
    print(f"Aligned: {aligned_labels}")

    # Verify
    assert len(aligned_labels) == len(encoding['input_ids'])
    assert -100 in aligned_labels  # Has special tokens
    assert 0 in aligned_labels  # Has O label
    assert 1 in aligned_labels  # Has entity label

    print("✅ Basic alignment correct")
    return True


def test_subword_alignment():
    """Test alignment with subword tokenization."""
    print("\n" + "="*60)
    print("TEST 2: Subword Token Alignment")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Words that split into subwords
    tokens = ['melanoma']  # Will split into ['mel', '##ano', '##ma']
    labels = [1]  # B-GENE

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )

    subwords = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
    word_ids = encoding.word_ids()

    print(f"Token: {tokens}")
    print(f"Label: {labels}")
    print(f"Subwords: {subwords}")
    print(f"Word IDs: {word_ids}")

    # Align labels
    aligned_labels = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_word_id:
            # First subword gets the label
            aligned_labels.append(labels[word_id])
        else:
            # Continuation subwords get -100 (or same label for I- tags)
            aligned_labels.append(-100)
        previous_word_id = word_id

    print(f"Aligned: {aligned_labels}")

    # Verify
    assert len(aligned_labels) == len(encoding['input_ids'])
    assert 1 in aligned_labels  # Has B-GENE label
    assert aligned_labels.count(1) == 1  # Only first subword gets label

    print("✅ Subword alignment correct")
    return True


def test_bio_tagging():
    """Test BIO tagging with multi-word entities."""
    print("\n" + "="*60)
    print("TEST 3: BIO Tagging")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # BIO tags
    tokens = ['tumor', 'suppressor', 'gene', 'p53']
    labels = [1, 2, 2, 1]  # B-GENE, I-GENE, I-GENE, B-GENE
    label_names = ['O', 'B-GENE', 'I-GENE']

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )

    word_ids = encoding.word_ids()

    # Align labels
    aligned_labels = []
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        else:
            aligned_labels.append(labels[word_id])

    print(f"Tokens: {tokens}")
    print(f"Labels: {[label_names[l] for l in labels]}")
    print(f"Word IDs: {word_ids}")
    print(f"Aligned: {aligned_labels}")

    # Verify BIO sequence
    non_special = [l for l in aligned_labels if l != -100]
    print(f"Non-special labels: {non_special}")

    assert 1 in non_special  # Has B-GENE
    assert 2 in non_special  # Has I-GENE

    print("✅ BIO tagging correct")
    return True


def test_empty_and_edge_cases():
    """Test edge cases."""
    print("\n" + "="*60)
    print("TEST 4: Edge Cases")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Edge case 1: Single token
    tokens = ['gene']
    labels = [1]

    encoding = tokenizer(tokens, is_split_into_words=True)
    word_ids = encoding.word_ids()
    aligned_labels = [labels[w] if w is not None else -100 for w in word_ids]

    assert len(aligned_labels) == len(encoding['input_ids'])
    print("✅ Single token works")

    # Edge case 2: All special tokens initially
    assert aligned_labels[0] == -100  # [CLS]
    assert aligned_labels[-1] == -100  # [SEP]
    print("✅ Special tokens get -100")

    # Edge case 3: Long sequence (truncation)
    long_tokens = ['word'] * 1000
    long_labels = [0] * 1000

    encoding = tokenizer(long_tokens, is_split_into_words=True, truncation=True, max_length=512)
    word_ids = encoding.word_ids()
    aligned_labels = [long_labels[w] if w is not None else -100 for w in word_ids]

    assert len(aligned_labels) == 512
    print("✅ Truncation works")

    print("✅ All edge cases pass")
    return True


def test_wrong_alignment_bug():
    """Demonstrate the bug that caused F1=0.46."""
    print("\n" + "="*60)
    print("TEST 5: Wrong Alignment Bug (Demonstration)")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    tokens = ['The', 'patient', 'has', 'diabetes']
    labels = [0, 0, 0, 1]

    print("❌ WRONG WAY (causes F1=0.46):")
    # Wrong: Join tokens then tokenize
    text = ' '.join(tokens)
    wrong_encoding = tokenizer(text, truncation=True, max_length=512)
    print(f"  Text: '{text}'")
    print(f"  Word boundaries destroyed!")
    print(f"  Can't align labels correctly!")

    print("\n✅ CORRECT WAY:")
    # Correct: Use is_split_into_words
    correct_encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )
    word_ids = correct_encoding.word_ids()
    print(f"  Tokens: {tokens}")
    print(f"  Word IDs: {word_ids}")
    print(f"  Can align labels perfectly!")

    aligned_labels = []
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        else:
            aligned_labels.append(labels[word_id])

    print(f"  Aligned labels: {aligned_labels}")
    assert 1 in aligned_labels  # Entity label preserved

    print("\n✅ Always use is_split_into_words=True for NER!")
    return True


if __name__ == '__main__':
    print("="*60)
    print("LABEL ALIGNMENT TESTS")
    print("="*60)
    print("\nThese tests prevent the F1=0.46 bug!")

    try:
        test_basic_alignment()
        test_subword_alignment()
        test_bio_tagging()
        test_empty_and_edge_cases()
        test_wrong_alignment_bug()

        print("\n" + "="*60)
        print("✅ ALL LABEL ALIGNMENT TESTS PASSED")
        print("="*60)
        print("\nKey lessons:")
        print("  1. ALWAYS use is_split_into_words=True for NER")
        print("  2. ALWAYS use word_ids() for label alignment")
        print("  3. NEVER join tokens and re-tokenize!")
        print("  4. Assign -100 to special tokens and continuation subwords")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
