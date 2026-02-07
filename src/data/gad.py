"""GAD dataset parser (Gene-Disease Association classification)."""

from pathlib import Path
from typing import Any, Dict, List, Literal
from datasets import load_from_disk

from .base import BaseTaskDataset, UnifiedSample, TaskRegistry


@TaskRegistry.register("gad")
class GADDataset(BaseTaskDataset):
    """GAD Gene-Disease Association dataset.

    Task: Binary classification - determine if gene-disease association exists
    Format: Text → Label (0=no association, 1=association)
    Size: 4,796 train / 534 test
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "dev", "test"],
        task_name: str = "gad",
    ):
        # GAD only has train/test, no validation
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
            task_type="classification",
            task_level="sequence",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Load GAD dataset from HuggingFace format."""
        dataset = load_from_disk(str(self.data_path / "gad"))

        raw_samples = []
        for sample in dataset[self.hf_split]:
            # GAD uses 'labels' (plural) which is a list like ['0'] or ['1']
            label_list = sample.get('labels', [])
            label = label_list[0] if label_list else '0'

            raw_samples.append({
                'id': sample['id'],
                'text': sample['text'],
                'label': label  # '0' or '1' as string
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
        """Convert GAD format to UnifiedSample."""
        text = raw_item['text']
        label = raw_item['label']  # 0 or 1

        # Convert label to string
        label_str = str(label)

        return UnifiedSample(
            task=self.task_name,
            task_type=self.task_type,
            task_level=self.task_level,
            input_text=text,
            labels=label_str,
            metadata={
                'id': raw_item['id'],
                'task_description': 'gene-disease association'
            },
            token_count=0  # Will be set during tokenization
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return binary label schema for GAD."""
        return {
            '0': 0,  # No association
            '1': 1   # Association exists
        }
