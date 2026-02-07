"""DDI dataset parser (Drug-Drug Interaction Relation Extraction)."""

from pathlib import Path
from typing import Any, Dict, List, Literal
from datasets import load_from_disk

from .base import BaseTaskDataset, UnifiedSample, TaskRegistry


@TaskRegistry.register("ddi")
class DDIDataset(BaseTaskDataset):
    """DDI Drug-Drug Interaction dataset.

    Task: Relation extraction between drug entities
    Relation types: DDI-mechanism, DDI-effect, DDI-advise, DDI-int
    Format: Text with drug entities and their interactions
    Size: 714 train / 303 test (no validation split)
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "dev", "test"],
        task_name: str = "ddi",
    ):
        # DDI only has train/test, no validation
        # Map dev → use part of train
        if split == "dev":
            self.hf_split = "train"
            self.use_dev_subset = True
        else:
            self.hf_split = split
            self.use_dev_subset = False

        super().__init__(
            data_path=data_path,
            split=split,
            task_name=task_name,
            task_type="re",
            task_level="pair",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Load DDI dataset from HuggingFace format."""
        dataset = load_from_disk(str(self.data_path / "ddi"))

        raw_samples = []
        for sample in dataset[self.hf_split]:
            # Extract text from passages
            text = ""
            if sample['passages']:
                text = sample['passages'][0]['text'][0] if isinstance(sample['passages'][0]['text'], list) else sample['passages'][0]['text']

            raw_samples.append({
                'id': sample['id'],
                'document_id': sample.get('document_id', ''),
                'text': text,
                'entities': sample['entities'],
                'relations': sample['relations']
            })

        # Create dev split from training data if needed
        if self.use_dev_subset:
            # Use last 20% of training data as dev
            split_idx = int(len(raw_samples) * 0.8)
            return raw_samples[split_idx:]
        elif self.split == "train" and self.hf_split == "train":
            # Use first 80% of training data
            split_idx = int(len(raw_samples) * 0.8)
            return raw_samples[:split_idx]
        else:
            return raw_samples

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert DDI format to UnifiedSample."""
        text = raw_item['text']
        entities = raw_item['entities']
        relations = raw_item['relations']

        # Format entities for easy lookup
        entity_dict = {e['id']: e for e in entities}

        # Extract relation triples
        relation_labels = []
        for rel in relations:
            rel_type = rel.get('type', 'UNKNOWN')
            arg1_id = rel.get('arg1_id', '')
            arg2_id = rel.get('arg2_id', '')

            # Get entity texts
            arg1_text = entity_dict.get(arg1_id, {}).get('text', [''])[0] if isinstance(entity_dict.get(arg1_id, {}).get('text', ['']), list) else entity_dict.get(arg1_id, {}).get('text', '')
            arg2_text = entity_dict.get(arg2_id, {}).get('text', [''])[0] if isinstance(entity_dict.get(arg2_id, {}).get('text', ['']), list) else entity_dict.get(arg2_id, {}).get('text', '')

            relation_labels.append((arg1_text, arg2_text, rel_type))

        return UnifiedSample(
            task=self.task_name,
            task_type=self.task_type,
            task_level=self.task_level,
            input_text=text,
            labels=relation_labels,  # List of (drug1, drug2, interaction_type) tuples
            metadata={
                'id': raw_item['id'],
                'document_id': raw_item['document_id'],
                'entities': entities,
                'num_relations': len(relations),
                'task_description': 'drug-drug interaction'
            },
            token_count=0  # Will be set during tokenization
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return relation type schema for DDI."""
        return {
            'DDI-mechanism': 0,  # Mechanism of interaction
            'DDI-effect': 1,     # Effect of interaction
            'DDI-advise': 2,     # Recommendation/advice
            'DDI-int': 3,        # General interaction
            'DDI-false': 4,      # No interaction (false positive)
            'NONE': 5            # No relation
        }
