"""JNLPBA dataset parser (Bio-entity NER with 5 entity types)."""

from pathlib import Path
from typing import Any, Dict, List, Literal
from datasets import load_from_disk

from .base import BaseTaskDataset, UnifiedSample, TaskRegistry


@TaskRegistry.register("jnlpba")
class JNLPBADataset(BaseTaskDataset):
    """JNLPBA Bio-entity Recognition dataset.

    Task: Named Entity Recognition for 5 bio-entity types
    Entities: DNA, RNA, Protein, Cell Line, Cell Type
    Format: BIO tagging with 11 tags (O + B/I for 5 entities)
    Size: 18,607 train / 1,939 val / 4,260 test
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "dev", "test"],
        task_name: str = "jnlpba",
    ):
        # JNLPBA uses "validation" instead of "dev"
        self.hf_split = "validation" if split == "dev" else split

        super().__init__(
            data_path=data_path,
            split=split,
            task_name=task_name,
            task_type="ner",
            task_level="token",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Load JNLPBA dataset from HuggingFace format."""
        dataset = load_from_disk(str(self.data_path / "jnlpba"))

        raw_samples = []
        for sample in dataset[self.hf_split]:
            raw_samples.append({
                'id': sample['id'],
                'tokens': sample['tokens'],
                'ner_tags': sample['ner_tags'],
            })

        return raw_samples

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert JNLPBA format to UnifiedSample."""
        tokens = raw_item['tokens']
        ner_tags = raw_item['ner_tags']

        # JNLPBA BIO tags:
        # 0=O, 1=B-DNA, 2=I-DNA, 3=B-RNA, 4=I-RNA,
        # 5=B-Protein, 6=I-Protein, 7=B-CellLine, 8=I-CellLine,
        # 9=B-CellType, 10=I-CellType
        tag_map = {
            0: 'O',
            1: 'B-DNA',
            2: 'I-DNA',
            3: 'B-RNA',
            4: 'I-RNA',
            5: 'B-Protein',
            6: 'I-Protein',
            7: 'B-CellLine',
            8: 'I-CellLine',
            9: 'B-CellType',
            10: 'I-CellType'
        }

        bio_labels = [tag_map.get(tag, 'O') for tag in ner_tags]

        return UnifiedSample(
            task=self.task_name,
            task_type=self.task_type,
            task_level=self.task_level,
            input_text=" ".join(tokens),
            labels=bio_labels,
            metadata={
                'id': raw_item['id'],
                'tokens': tokens,
                'entity_types': ['DNA', 'RNA', 'Protein', 'CellLine', 'CellType']
            },
            token_count=len(tokens)
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return BIO label schema for JNLPBA (5 entity types)."""
        return {
            'O': 0,
            'B-DNA': 1,
            'I-DNA': 2,
            'B-RNA': 3,
            'I-RNA': 4,
            'B-Protein': 5,
            'I-Protein': 6,
            'B-CellLine': 7,
            'I-CellLine': 8,
            'B-CellType': 9,
            'I-CellType': 10
        }
