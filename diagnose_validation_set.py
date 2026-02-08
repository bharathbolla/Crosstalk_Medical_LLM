"""
Diagnostic script to check validation set issues causing perfect 1.0 scores.
Add this cell RIGHT AFTER loading the pickle data in Kaggle.
"""

import pickle
from pathlib import Path
from collections import Counter

def diagnose_validation_set(dataset_name, raw_data, val_split='validation'):
    """Diagnose validation set to identify perfect 1.0 score causes."""

    print(f"\n{'='*60}")
    print(f"🔍 DIAGNOSING VALIDATION SET: {dataset_name.upper()}")
    print(f"{'='*60}")

    train_samples = raw_data['train']
    val_samples = raw_data[val_split]

    print(f"\n1️⃣ Sample Counts:")
    print(f"   Train: {len(train_samples)}")
    print(f"   Val: {len(val_samples)}")

    # Check label distribution in validation set
    print(f"\n2️⃣ Validation Label Distribution:")
    val_labels = []
    for sample in val_samples:
        label = sample.get('label', sample.get('relation', sample.get('ner_tags', None)))
        if label is not None:
            val_labels.append(label)

    if val_labels:
        label_counts = Counter(val_labels)
        print(f"   Total labels: {len(val_labels)}")
        print(f"   Unique labels: {len(label_counts)}")
        print(f"   Distribution:")
        for label, count in label_counts.most_common(10):
            percentage = (count / len(val_labels)) * 100
            print(f"      {label}: {count} ({percentage:.1f}%)")

        # Check if highly imbalanced
        most_common = label_counts.most_common(1)[0]
        if most_common[1] >= len(val_labels) * 0.90:
            print(f"\n   ⚠️  WARNING: {most_common[1]/len(val_labels)*100:.1f}% of validation samples have label '{most_common[0]}'")
            print(f"   This extreme imbalance can cause perfect 1.0 scores!")

    # Check for text overlap between train and val
    print(f"\n3️⃣ Checking for Data Leakage:")
    train_texts = set()
    for sample in train_samples:
        text = sample.get('text', '')
        if text:
            # Use first 100 chars as fingerprint
            train_texts.add(text[:100])

    val_texts = set()
    overlapping = 0
    for sample in val_samples:
        text = sample.get('text', '')
        if text:
            fingerprint = text[:100]
            val_texts.add(fingerprint)
            if fingerprint in train_texts:
                overlapping += 1

    print(f"   Unique train texts: {len(train_texts)}")
    print(f"   Unique val texts: {len(val_texts)}")
    print(f"   Overlapping texts: {overlapping}")

    if overlapping > 0:
        percentage = (overlapping / len(val_samples)) * 100
        print(f"\n   ⚠️  DATA LEAKAGE: {percentage:.1f}% of validation samples overlap with training!")
        print(f"   This causes perfect 1.0 scores!")

    # Check for empty texts
    print(f"\n4️⃣ Checking for Empty Texts:")
    empty_train = sum(1 for s in train_samples if not s.get('text', '').strip())
    empty_val = sum(1 for s in val_samples if not s.get('text', '').strip())

    print(f"   Empty train texts: {empty_train}")
    print(f"   Empty val texts: {empty_val}")

    if empty_val > 0:
        print(f"\n   ⚠️  WARNING: {empty_val} validation samples have empty text!")
        print(f"   This causes model to learn trivial patterns!")

    # Print sample texts
    print(f"\n5️⃣ Sample Validation Texts:")
    for i, sample in enumerate(val_samples[:3], 1):
        text = sample.get('text', '')[:200]
        label = sample.get('label', sample.get('relation', 'N/A'))
        print(f"\n   Sample {i}:")
        print(f"   Label: {label}")
        print(f"   Text: {text}...")

    print(f"\n{'='*60}")
    print(f"DIAGNOSIS COMPLETE")
    print(f"{'='*60}\n")


# Example usage - add this in your Kaggle notebook after loading pickle data:
"""
# After line 641 in the multi-dataset loop, add:
exec(open('diagnose_validation_set.py').read())
diagnose_validation_set(primary_dataset, raw_data, val_split)
"""
