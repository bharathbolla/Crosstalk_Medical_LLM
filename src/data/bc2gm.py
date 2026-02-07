"""BC2GM dataset parser (Gene/Protein NER)."""

from pathlib import Path
from typing import Any, Dict, List, Literal
from datasets import load_from_disk

from .base import BaseTaskDataset, UnifiedSample, TaskRegistry


@TaskRegistry.register("bc2gm")
class BC2GMDataset(BaseTaskDataset):
    """BC2GM Gene Mention Recognition dataset.

    Task: Named Entity Recognition for genes and proteins
    Format: BIO tagging (0=O, 1=B-Gene, 2=I-Gene)
    Size: 12,574 train / 2,519 val / 5,038 test
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "dev", "test"],
        task_name: str = "bc2gm",
    ):
        # BC2GM uses "validation" instead of "dev"
        self.hf_split = "validation" if split == "dev" else split

        super().__init__(
            data_path=data_path,
            split=split,
            task_name=task_name,
            task_type="ner",
            task_level="token",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Load BC2GM dataset from HuggingFace format."""
        # Load from disk (downloaded via download_diverse_tasks.py)
        dataset = load_from_disk(str(self.data_path / "bc2gm"))

        raw_samples = []
        for sample in dataset[self.hf_split]:
            raw_samples.append({
                'id': sample['id'],
                'tokens': sample['tokens'],
                'ner_tags': sample['ner_tags'],
            })

        return raw_samples

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert BC2GM format to UnifiedSample."""
        tokens = raw_item['tokens']
        ner_tags = raw_item['ner_tags']

        # Convert BIO tags to labels (keep as BIO for training)
        # 0=O, 1=B-Gene, 2=I-Gene
        bio_labels = []
        for tag in ner_tags:
            if tag == 0:
                bio_labels.append('O')
            elif tag == 1:
                bio_labels.append('B-Gene')
            elif tag == 2:
                bio_labels.append('I-Gene')
            else:
                bio_labels.append('O')  # Fallback

        return UnifiedSample(
            task=self.task_name,
            task_type=self.task_type,
            task_level=self.task_level,
            input_text=" ".join(tokens),
            labels=bio_labels,
            metadata={
                'id': raw_item['id'],
                'tokens': tokens,
                'entity_type': 'Gene'
            },
            token_count=len(tokens)
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return BIO label schema for BC2GM."""
        return {
            'O': 0,
            'B-Gene': 1,
            'I-Gene': 2
        }
