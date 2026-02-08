"""
Debug script to understand why ChemProt/DDI get perfect 1.0 scores
Run this in Kaggle to diagnose the issue
"""

import pickle
import torch
from pathlib import Path
from collections import Counter

print("="*60)
print("DEBUGGING PERFECT SCORES")
print("="*60)

# Load ChemProt
print("\n1. Checking ChemProt data...")
with open('data/pickle/chemprot.pkl', 'rb') as f:
    chemprot = pickle.load(f)

# Check validation split
val_split = 'validation' if 'validation' in chemprot else 'test'
print(f"   Using split: {val_split}")
print(f"   Train docs: {len(chemprot['train'])}")
print(f"   Val docs: {len(chemprot[val_split])}")

# Limit to first 100 docs for train, 20 for val
train_docs = chemprot['train'][:100]
val_docs = chemprot[val_split][:20]

print(f"\n2. After limiting:")
print(f"   Train docs: {len(train_docs)}")
print(f"   Val docs: {len(val_docs)}")

# Count relations (actual training samples)
train_relations = []
for doc in train_docs:
    for rel in doc.get('relations', []):
        train_relations.append(rel.get('type', rel.get('label')))

val_relations = []
for doc in val_docs:
    for rel in doc.get('relations', []):
        val_relations.append(rel.get('type', rel.get('label')))

print(f"\n3. Actual relation samples:")
print(f"   Train relations: {len(train_relations)}")
print(f"   Val relations: {len(val_relations)}")

print(f"\n4. Label distribution:")
print(f"   Train labels: {Counter(train_relations)}")
print(f"   Val labels: {Counter(val_relations)}")

# Check if validation set has only one label
unique_val_labels = set(val_relations)
print(f"\n5. Unique labels in validation: {len(unique_val_labels)}")
if len(unique_val_labels) <= 2:
    print(f"   ⚠️  WARNING: Only {len(unique_val_labels)} unique labels in validation!")
    print(f"   Labels: {unique_val_labels}")
    print(f"   This could cause perfect 1.0 scores if model predicts majority class")

# Check for data leakage at document level
train_doc_ids = set(d.get('id', d.get('document_id')) for d in train_docs)
val_doc_ids = set(d.get('id', d.get('document_id')) for d in val_docs)
overlap = train_doc_ids.intersection(val_doc_ids)

print(f"\n6. Document-level overlap:")
print(f"   Overlapping doc IDs: {len(overlap)}")
if overlap:
    print(f"   ⚠️  DATA LEAKAGE: {overlap}")

# Check if all val samples are positive class
if len(val_relations) > 0:
    most_common_label = Counter(val_relations).most_common(1)[0]
    print(f"\n7. Most common val label: {most_common_label[0]} ({most_common_label[1]}/{len(val_relations)} = {most_common_label[1]/len(val_relations)*100:.1f}%)")
    if most_common_label[1] == len(val_relations):
        print(f"   ⚠️  ALL validation samples have the same label!")
        print(f"   This explains perfect 1.0 scores")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
