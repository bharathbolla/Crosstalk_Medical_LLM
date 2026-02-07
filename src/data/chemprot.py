"""ChemProt dataset parser (Chemical-Protein Relation Extraction)."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple
from datasets import load_from_disk

from .base import BaseTaskDataset, UnifiedSample, TaskRegistry


@TaskRegistry.register("chemprot")
class ChemProtDataset(BaseTaskDataset):
    """ChemProt Chemical-Protein Interaction dataset.

    Task: Relation extraction between chemical and protein entities
    Relation types: CPR:3 (UPREGULATOR), CPR:4 (DOWNREGULATOR),
                    CPR:5 (AGONIST), CPR:6 (ANTAGONIST), CPR:9 (SUBSTRATE)
    Format: Text with entities and relations
    Size: 1,020 train / 612 val / 800 test
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "dev", "test"],
        task_name: str = "chemprot",
    ):
        # ChemProt uses "validation" instead of "dev"
        self.hf_split = "validation" if split == "dev" else split

        super().__init__(
            data_path=data_path,
            split=split,
            task_name=task_name,
            task_type="re",
            task_level="pair",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Load ChemProt dataset from HuggingFace format."""
        dataset = load_from_disk(str(self.data_path / "chemprot"))

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

        return raw_samples

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert ChemProt format to UnifiedSample."""
        text = raw_item['text']
        entities = raw_item['entities']
        relations = raw_item['relations']

        # Format entities for easy lookup
        entity_dict = {e['id']: e for e in entities}

        # Extract relation triples (head_entity, tail_entity, relation_type)
        relation_labels = []
        for rel in relations:
            rel_type = rel.get('type', 'UNKNOWN')
            arg1_id = rel.get('arg1_id', '')
            arg2_id = rel.get('arg2_id', '')

            # Get entity texts
            arg1_text = entity_dict.get(arg1_id, {}).get('text', [''])[0] if isinstance(entity_dict.get(arg1_id, {}).get('text', ['']), list) else entity_dict.get(arg1_id, {}).get('text', '')
            arg2_text = entity_dict.get(arg2_id, {}).get('text', [''])[0] if isinstance(entity_dict.get(arg2_id, {}).get('text', ['']), list) else entity_dict.get(arg2_id, {}).get('text', '')

            relation_labels.append((arg1_text, arg2_text, rel_type))

        # For input, mark entities in text
        # Simple format: just use original text
        # The model will learn to identify entity pairs

        return UnifiedSample(
            task=self.task_name,
            task_type=self.task_type,
            task_level=self.task_level,
            input_text=text,
            labels=relation_labels,  # List of (entity1, entity2, relation) tuples
            metadata={
                'id': raw_item['id'],
                'document_id': raw_item['document_id'],
                'entities': entities,
                'num_relations': len(relations),
                'task_description': 'chemical-protein interaction'
            },
            token_count=0  # Will be set during tokenization
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return relation type schema for ChemProt."""
        return {
            'CPR:3': 0,  # UPREGULATOR
            'CPR:4': 1,  # DOWNREGULATOR
            'CPR:5': 2,  # AGONIST
            'CPR:6': 3,  # ANTAGONIST
            'CPR:9': 4,  # SUBSTRATE
            'NONE': 5    # No relation
        }
