"""
Regenerate pickle files in the CORRECT flat format for UniversalMedicalDataset.

This script:
1. Loads raw datasets from data/raw/
2. Processes them with dataset-specific parsers
3. Flattens documents into individual samples (for RE tasks)
4. Saves in format: {'text': ..., 'label': ..., 'tokens': ...}

Run this BEFORE uploading to Kaggle!
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pickle
from pathlib import Path
from datasets import load_from_disk

def process_ner_dataset(dataset, dataset_name):
    """Process NER datasets (BC2GM, JNLPBA) - already flat."""
    processed = {}

    for split_name in dataset.keys():
        samples = []
        for example in dataset[split_name]:
            samples.append({
                'tokens': example['tokens'],
                'ner_tags': example['ner_tags'],
                'id': example.get('id', ''),
            })
        processed[split_name] = samples
        print(f"   {split_name}: {len(samples)} samples")

    return processed


def process_re_dataset(dataset, dataset_name):
    """Process RE datasets (ChemProt, DDI) - FLATTEN documents to individual relations."""
    processed = {}

    for split_name in dataset.keys():
        samples = []

        for doc in dataset[split_name]:
            # Extract text from passages
            text = ""
            if doc.get('passages'):
                passage = doc['passages'][0]
                text = passage['text'][0] if isinstance(passage['text'], list) else passage['text']

            # Get relations
            relations = doc.get('relations', [])
            entities = {e['id']: e for e in doc.get('entities', [])}

            if not relations:
                # Document with no relations - skip or add negative sample
                continue

            # Flatten: One sample per relation
            for rel in relations:
                rel_type = rel.get('type', 'NONE')
                arg1_id = rel.get('arg1_id', '')
                arg2_id = rel.get('arg2_id', '')

                # Get entity texts
                arg1 = entities.get(arg1_id, {})
                arg2 = entities.get(arg2_id, {})

                arg1_text = arg1.get('text', [''])[0] if isinstance(arg1.get('text'), list) else arg1.get('text', '')
                arg2_text = arg2.get('text', [''])[0] if isinstance(arg2.get('text'), list) else arg2.get('text', '')

                # Create flattened sample with text and label
                samples.append({
                    'text': text,
                    'entity1': arg1_text,
                    'entity2': arg2_text,
                    'relation': rel_type,
                    'label': rel_type,  # Both 'relation' and 'label' for compatibility
                    'id': doc.get('id', ''),
                    'document_id': doc.get('document_id', ''),
                })

        processed[split_name] = samples
        print(f"   {split_name}: {len(samples)} relation samples (flattened from documents)")

    return processed


def process_classification_dataset(dataset, dataset_name):
    """Process classification datasets (GAD, HoC, PubMedQA)."""
    processed = {}

    for split_name in dataset.keys():
        samples = []

        for example in dataset[split_name]:
            # Extract text
            if 'question' in example:  # PubMedQA
                text = f"{example['question']} {example.get('context', '')}"
                label = example.get('final_decision', example.get('label', ''))
            elif 'text' in example:  # GAD
                text = example['text']
                label = example.get('label', 0)
            elif 'passages' in example:  # HoC
                passage = example['passages'][0]
                text = passage['text'][0] if isinstance(passage['text'], list) else passage['text']
                label = example.get('label', [])
            else:
                text = str(example)
                label = example.get('label', 0)

            samples.append({
                'text': text,
                'label': label,
                'id': example.get('id', ''),
            })

        processed[split_name] = samples
        print(f"   {split_name}: {len(samples)} samples")

    return processed


def process_similarity_dataset(dataset, dataset_name):
    """Process similarity dataset (BIOSSES)."""
    processed = {}

    for split_name in dataset.keys():
        samples = []

        for example in dataset[split_name]:
            samples.append({
                'sentence1': example.get('sentence1', example.get('text1', '')),
                'sentence2': example.get('sentence2', example.get('text2', '')),
                'score': float(example.get('score', example.get('label', 0.0))),
                'label': float(example.get('score', example.get('label', 0.0))),  # For compatibility
                'id': example.get('id', ''),
            })

        processed[split_name] = samples
        print(f"   {split_name}: {len(samples)} pairs")

    return processed


def main():
    """Regenerate all pickle files in correct flat format."""

    data_path = Path("data/raw")
    output_path = Path("data/pickle")
    output_path.mkdir(parents=True, exist_ok=True)

    # Define dataset types
    ner_datasets = ["bc2gm", "jnlpba"]
    re_datasets = ["chemprot", "ddi"]
    classification_datasets = ["gad", "hoc", "pubmedqa"]
    similarity_datasets = ["biosses"]

    print("=" * 60)
    print("REGENERATING PICKLE FILES (FLAT FORMAT)")
    print("=" * 60)

    # Process NER datasets
    for dataset_name in ner_datasets:
        print(f"\n📦 {dataset_name.upper()} (NER)")
        try:
            dataset = load_from_disk(str(data_path / dataset_name))
            processed = process_ner_dataset(dataset, dataset_name)

            output_file = output_path / f"{dataset_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(processed, f)
            print(f"   ✅ Saved to {output_file}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Process RE datasets (CRITICAL - these need flattening!)
    for dataset_name in re_datasets:
        print(f"\n📦 {dataset_name.upper()} (RE - FLATTENING)")
        try:
            dataset = load_from_disk(str(data_path / dataset_name))
            processed = process_re_dataset(dataset, dataset_name)

            output_file = output_path / f"{dataset_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(processed, f)
            print(f"   ✅ Saved to {output_file}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Process classification datasets
    for dataset_name in classification_datasets:
        print(f"\n📦 {dataset_name.upper()} (Classification)")
        try:
            dataset = load_from_disk(str(data_path / dataset_name))
            processed = process_classification_dataset(dataset, dataset_name)

            output_file = output_path / f"{dataset_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(processed, f)
            print(f"   ✅ Saved to {output_file}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    # Process similarity datasets
    for dataset_name in similarity_datasets:
        print(f"\n📦 {dataset_name.upper()} (Similarity)")
        try:
            dataset = load_from_disk(str(data_path / dataset_name))
            processed = process_similarity_dataset(dataset, dataset_name)

            output_file = output_path / f"{dataset_name}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(processed, f)
            print(f"   ✅ Saved to {output_file}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n" + "=" * 60)
    print("✅ REGENERATION COMPLETE!")
    print("=" * 60)
    print("\nWhat changed:")
    print("  - ChemProt & DDI: FLATTENED documents → individual relation samples")
    print("  - All datasets: Now have 'text' and 'label' fields")
    print("  - Ready for UniversalMedicalDataset!")
    print("\nNext steps:")
    print("  1. Commit new pickle files")
    print("  2. Upload to Kaggle")
    print("  3. Re-run multi-dataset test")
    print("  4. Expect REALISTIC scores (not perfect 1.0)!")


if __name__ == "__main__":
    main()
