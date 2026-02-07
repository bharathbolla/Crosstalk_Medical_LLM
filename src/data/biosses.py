"""BIOSSES dataset parser (Biomedical Sentence Similarity)."""

from pathlib import Path
from typing import Any, Dict, List, Literal
from datasets import load_from_disk

from .base import BaseTaskDataset, UnifiedSample, TaskRegistry


@TaskRegistry.register("biosses")
class BIOSSESDataset(BaseTaskDataset):
    """BIOSSES Biomedical Sentence Similarity dataset.

    Task: Predict semantic similarity between sentence pairs
    Format: Sentence pair → Similarity score (0-4)
    Size: 64 train / 16 val / 20 test (small dataset!)
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "dev", "test"],
        task_name: str = "biosses",
    ):
        # BIOSSES uses "validation" instead of "dev"
        self.hf_split = "validation" if split == "dev" else split

        super().__init__(
            data_path=data_path,
            split=split,
            task_name=task_name,
            task_type="classification",  # Treat as classification (binned similarity)
            task_level="sequence",  # Sequence-level classification (not pair-level)
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Load BIOSSES dataset from HuggingFace format."""
        dataset = load_from_disk(str(self.data_path / "biosses"))

        raw_samples = []
        for sample in dataset[self.hf_split]:
            raw_samples.append({
                'id': sample['id'],
                'document_id': sample.get('document_id', ''),
                'text_1': sample['text_1'],
                'text_2': sample['text_2'],
                'label': sample['label']  # Similarity score (float 0-4)
            })

        return raw_samples

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert BIOSSES format to UnifiedSample."""
        text_1 = raw_item['text_1']
        text_2 = raw_item['text_2']
        similarity = float(raw_item['label'])  # Convert to float

        # Format as [S1] sentence1 [S2] sentence2
        input_text = f"[S1] {text_1} [S2] {text_2}"

        # Bin similarity score into 5 categories (0-1, 1-2, 2-3, 3-4, 4)
        # Or keep as continuous value for regression
        # For classification, we'll bin it:
        if similarity < 1.0:
            label = '0'  # Very dissimilar
        elif similarity < 2.0:
            label = '1'  # Dissimilar
        elif similarity < 3.0:
            label = '2'  # Somewhat similar
        elif similarity < 4.0:
            label = '3'  # Similar
        else:
            label = '4'  # Very similar

        return UnifiedSample(
            task=self.task_name,
            task_type=self.task_type,
            task_level=self.task_level,
            input_text=input_text,
            labels=label,
            metadata={
                'id': raw_item['id'],
                'document_id': raw_item['document_id'],
                'text_1': text_1,
                'text_2': text_2,
                'similarity_score': float(similarity),  # Keep original score
                'task_description': 'sentence similarity'
            },
            token_count=0  # Will be set during tokenization
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return label schema for BIOSSES (5 similarity bins)."""
        return {
            '0': 0,  # Very dissimilar (0-1)
            '1': 1,  # Dissimilar (1-2)
            '2': 2,  # Somewhat similar (2-3)
            '3': 3,  # Similar (3-4)
            '4': 4   # Very similar (4)
        }
