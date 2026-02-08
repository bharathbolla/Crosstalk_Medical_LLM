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
from collections import Counter
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
                # Document with no relations - skip
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
                    'label': rel_type,
                    'id': doc.get('id', ''),
                    'document_id': doc.get('document_id', ''),
                })

        processed[split_name] = samples
        label_dist = Counter(s['label'] for s in samples)
        print(f"   {split_name}: {len(samples)} relation samples (flattened)")
        print(f"      Labels: {dict(label_dist.most_common())}")

    return processed


def process_gad_dataset(dataset):
    """Process GAD (Gene-Disease Association) - binary classification.

    Raw format: {'text': ..., 'labels': ['0'] or ['1'], ...}
    Note: key is 'labels' (plural, list of strings), NOT 'label'.
    """
    processed = {}

    for split_name in dataset.keys():
        samples = []
        for example in dataset[split_name]:
            text = example['text']
            # Raw data uses 'labels' (plural) as a list of strings
            labels_list = example.get('labels', example.get('label', ['0']))
            if isinstance(labels_list, list):
                label = int(labels_list[0])
            else:
                label = int(labels_list)

            samples.append({
                'text': text,
                'label': label,
                'id': example.get('id', ''),
            })

        processed[split_name] = samples
        label_dist = Counter(s['label'] for s in samples)
        print(f"   {split_name}: {len(samples)} samples")
        print(f"      Labels: {dict(label_dist)}")

    return processed


def process_hoc_dataset(dataset):
    """Process HoC (Hallmarks of Cancer) - multi-label classification.

    Raw format: {'text': ..., 'labels': ['activating invasion and metastasis', ...], ...}
    Note: key is 'labels' (plural), list of hallmark names.
    """
    # First pass: collect all unique labels
    all_labels = set()
    for split_name in dataset.keys():
        for example in dataset[split_name]:
            labels_list = example.get('labels', [])
            for lbl in labels_list:
                if lbl.lower() != 'none':
                    all_labels.add(lbl)

    all_labels = sorted(all_labels)
    label2id = {lbl: idx for idx, lbl in enumerate(all_labels)}
    print(f"   HoC unique labels ({len(all_labels)}): {all_labels}")

    processed = {}
    for split_name in dataset.keys():
        samples = []
        for example in dataset[split_name]:
            text = example['text']
            labels_list = example.get('labels', [])

            # Convert to multi-hot encoding
            multi_hot = [0] * len(all_labels)
            for lbl in labels_list:
                if lbl.lower() != 'none' and lbl in label2id:
                    multi_hot[label2id[lbl]] = 1

            samples.append({
                'text': text,
                'label': multi_hot,
                'label_names': [lbl for lbl in labels_list if lbl.lower() != 'none'],
                'id': example.get('id', ''),
            })

        processed[split_name] = samples
        # Count label frequency
        label_counts = Counter()
        for s in samples:
            for name in s['label_names']:
                label_counts[name] += 1
        print(f"   {split_name}: {len(samples)} samples")
        print(f"      Top labels: {dict(label_counts.most_common(5))}")

    # Store label list in metadata
    processed['_label_names'] = all_labels
    return processed


def process_pubmedqa_dataset(dataset):
    """Process PubMedQA - 3-class classification (yes/no/maybe).

    Raw format: {'question': ..., 'context': {'contexts': [...], ...}, 'final_decision': 'yes'}
    Note: context is a dict with 'contexts' list, NOT a plain string.
    """
    processed = {}

    for split_name in dataset.keys():
        samples = []
        for example in dataset[split_name]:
            question = example.get('question', '')

            # Context is a dict with 'contexts' key containing a list
            context_data = example.get('context', {})
            if isinstance(context_data, dict):
                contexts = context_data.get('contexts', [])
                context_text = ' '.join(contexts) if contexts else ''
            elif isinstance(context_data, str):
                context_text = context_data
            else:
                context_text = str(context_data)

            text = f"{question} {context_text}".strip()
            label = example.get('final_decision', '')

            samples.append({
                'text': text,
                'label': label,
                'id': example.get('pubid', example.get('id', '')),
            })

        processed[split_name] = samples
        label_dist = Counter(s['label'] for s in samples)
        print(f"   {split_name}: {len(samples)} samples")
        print(f"      Labels: {dict(label_dist)}")

    return processed


def process_biosses_dataset(dataset):
    """Process BIOSSES - sentence similarity (regression).

    Raw format: {'text_1': ..., 'text_2': ..., 'label': '4.0', ...}
    Note: keys are 'text_1'/'text_2' (with underscore), NOT 'sentence1'/'text1'.
    """
    processed = {}

    for split_name in dataset.keys():
        samples = []
        for example in dataset[split_name]:
            # Raw data uses text_1 and text_2 (with underscore)
            sentence1 = example.get('text_1', example.get('sentence1', ''))
            sentence2 = example.get('text_2', example.get('sentence2', ''))
            score = float(example.get('label', example.get('score', 0.0)))

            samples.append({
                'sentence1': sentence1,
                'sentence2': sentence2,
                'text': f"{sentence1} [SEP] {sentence2}",
                'score': score,
                'label': score,
                'id': example.get('id', ''),
            })

        processed[split_name] = samples
        scores = [s['score'] for s in samples]
        print(f"   {split_name}: {len(samples)} pairs")
        if scores:
            print(f"      Score range: {min(scores):.1f} - {max(scores):.1f}, mean: {sum(scores)/len(scores):.2f}")

    return processed


def main():
    """Regenerate all pickle files in correct flat format."""

    data_path = Path("data/raw")
    output_path = Path("data/pickle")
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("REGENERATING PICKLE FILES (FLAT FORMAT)")
    print("=" * 60)

    # ---- NER datasets (already flat) ----
    for dataset_name in ["bc2gm", "jnlpba"]:
        print(f"\n--- {dataset_name.upper()} (NER) ---")
        try:
            dataset = load_from_disk(str(data_path / dataset_name))
            processed = process_ner_dataset(dataset, dataset_name)
            with open(output_path / f"{dataset_name}.pkl", 'wb') as f:
                pickle.dump(processed, f)
            print(f"   Saved!")
        except Exception as e:
            print(f"   ERROR: {e}")

    # ---- RE datasets (need flattening) ----
    for dataset_name in ["chemprot", "ddi"]:
        print(f"\n--- {dataset_name.upper()} (RE - FLATTENING) ---")
        try:
            dataset = load_from_disk(str(data_path / dataset_name))
            processed = process_re_dataset(dataset, dataset_name)
            with open(output_path / f"{dataset_name}.pkl", 'wb') as f:
                pickle.dump(processed, f)
            print(f"   Saved!")
        except Exception as e:
            print(f"   ERROR: {e}")

    # ---- GAD (binary classification) ----
    print(f"\n--- GAD (Binary Classification) ---")
    try:
        dataset = load_from_disk(str(data_path / "gad"))
        processed = process_gad_dataset(dataset)
        with open(output_path / "gad.pkl", 'wb') as f:
            pickle.dump(processed, f)
        print(f"   Saved!")
    except Exception as e:
        print(f"   ERROR: {e}")

    # ---- HoC (multi-label classification) ----
    print(f"\n--- HOC (Multi-label Classification) ---")
    try:
        dataset = load_from_disk(str(data_path / "hoc"))
        processed = process_hoc_dataset(dataset)
        with open(output_path / "hoc.pkl", 'wb') as f:
            pickle.dump(processed, f)
        print(f"   Saved!")
    except Exception as e:
        print(f"   ERROR: {e}")

    # ---- PubMedQA (3-class classification) ----
    print(f"\n--- PUBMEDQA (QA Classification) ---")
    try:
        dataset = load_from_disk(str(data_path / "pubmedqa"))
        processed = process_pubmedqa_dataset(dataset)
        with open(output_path / "pubmedqa.pkl", 'wb') as f:
            pickle.dump(processed, f)
        print(f"   Saved!")
    except Exception as e:
        print(f"   ERROR: {e}")

    # ---- BIOSSES (similarity/regression) ----
    print(f"\n--- BIOSSES (Similarity/Regression) ---")
    try:
        dataset = load_from_disk(str(data_path / "biosses"))
        processed = process_biosses_dataset(dataset)
        with open(output_path / "biosses.pkl", 'wb') as f:
            pickle.dump(processed, f)
        print(f"   Saved!")
    except Exception as e:
        print(f"   ERROR: {e}")

    print("\n" + "=" * 60)
    print("REGENERATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
