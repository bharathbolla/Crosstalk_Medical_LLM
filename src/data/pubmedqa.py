"""PubMedQA dataset parser (Medical Question Answering)."""

from pathlib import Path
from typing import Any, Dict, List, Literal
from datasets import load_from_disk

from .base import BaseTaskDataset, UnifiedSample, TaskRegistry


@TaskRegistry.register("pubmedqa")
class PubMedQADataset(BaseTaskDataset):
    """PubMedQA Medical Question Answering dataset.

    Task: Answer research questions with yes/no/maybe
    Format: Question + Context → Answer
    Size: 1,000 labeled examples
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "dev", "test"],
        task_name: str = "pubmedqa",
    ):
        # PubMedQA only has "train" split in labeled version
        # We'll use it all for training and create dev/test splits if needed
        self.hf_split = "train"

        super().__init__(
            data_path=data_path,
            split=split,
            task_name=task_name,
            task_type="qa",
            task_level="sequence",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Load PubMedQA dataset from HuggingFace format."""
        dataset = load_from_disk(str(self.data_path / "pubmedqa"))

        raw_samples = []
        for sample in dataset[self.hf_split]:
            # Extract context (list of sentences)
            contexts = sample.get('context', {}).get('contexts', [])
            context_text = " ".join(contexts) if contexts else ""

            raw_samples.append({
                'pubid': sample.get('pubid', ''),
                'question': sample['question'],
                'context': context_text,
                'long_answer': sample.get('long_answer', ''),
                'final_decision': sample['final_decision']  # yes/no/maybe
            })

        # Split dataset if needed (since only train exists)
        if self.split == "train":
            return raw_samples[:800]  # 80% for training
        elif self.split == "dev":
            return raw_samples[800:900]  # 10% for dev
        else:  # test
            return raw_samples[900:]  # 10% for test

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert PubMedQA format to UnifiedSample."""
        question = raw_item['question']
        context = raw_item['context']
        answer = raw_item['final_decision']  # yes/no/maybe

        # Format as [Q] question [C] context
        input_text = f"[Q] {question} [C] {context}"

        return UnifiedSample(
            task=self.task_name,
            task_type=self.task_type,
            task_level=self.task_level,
            input_text=input_text,
            labels=answer,  # Single label: yes/no/maybe
            metadata={
                'pubid': raw_item['pubid'],
                'question': question,
                'context': context,
                'long_answer': raw_item['long_answer']
            },
            token_count=0  # Will be set during tokenization
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return label schema for PubMedQA (3-way classification)."""
        return {
            'yes': 0,
            'no': 1,
            'maybe': 2
        }
