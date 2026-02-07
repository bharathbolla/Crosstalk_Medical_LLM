"""HoC dataset parser (Hallmarks of Cancer multi-label classification)."""

from pathlib import Path
from typing import Any, Dict, List, Literal
from datasets import load_from_disk

from .base import BaseTaskDataset, UnifiedSample, TaskRegistry


@TaskRegistry.register("hoc")
class HoCDataset(BaseTaskDataset):
    """HoC Hallmarks of Cancer dataset.

    Task: Multi-label document classification
    Labels: 10 cancer hallmark categories
    Format: Abstract → Multiple binary labels
    Size: 12,119 train / 1,798 val / 3,547 test
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "dev", "test"],
        task_name: str = "hoc",
    ):
        # HoC uses "validation" instead of "dev"
        self.hf_split = "validation" if split == "dev" else split

        super().__init__(
            data_path=data_path,
            split=split,
            task_name=task_name,
            task_type="classification",
            task_level="sequence",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Load HoC dataset from HuggingFace format."""
        dataset = load_from_disk(str(self.data_path / "hoc"))

        raw_samples = []
        for sample in dataset[self.hf_split]:
            # HoC has multi-label format
            raw_samples.append({
                'id': sample['id'],
                'document_id': sample.get('document_id', ''),
                'text': sample['text'],
                'labels': sample['labels']  # List of label indices
            })

        return raw_samples

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert HoC format to UnifiedSample."""
        text = raw_item['text']
        labels = raw_item['labels']  # List of integers

        # Convert to binary vector or keep as list
        # For multi-label, we keep as list of label indices
        label_str = ','.join(map(str, sorted(labels)))

        return UnifiedSample(
            task=self.task_name,
            task_type=self.task_type,
            task_level=self.task_level,
            input_text=text,
            labels=label_str,  # Store as comma-separated string
            metadata={
                'id': raw_item['id'],
                'document_id': raw_item['document_id'],
                'label_list': labels,  # Keep original list
                'num_labels': len(labels),
                'task_description': 'hallmarks of cancer'
            },
            token_count=0  # Will be set during tokenization
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return label schema for HoC (10 hallmark categories).

        The 10 hallmarks are:
        0: Sustaining proliferative signaling
        1: Evading growth suppressors
        2: Resisting cell death
        3: Enabling replicative immortality
        4: Inducing angiogenesis
        5: Activating invasion and metastasis
        6: Genomic instability and mutation
        7: Tumor promoting inflammation
        8: Cellular energetics
        9: Avoiding immune destruction
        """
        return {str(i): i for i in range(10)}
