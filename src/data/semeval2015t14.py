"""SemEval-2015 Task 14: Analysis of Clinical Text (Disorders).

Task: Disorder identification with **discontiguous entities**
Format: BRAT annotation with multiple spans per entity
Subtasks:
  A: Disorder span detection (including discontinuous)
  B: Normalization to SNOMED-CT

We implement Subtask A using span-based format (NOT BIO).

Data source: https://alt.qcri.org/semeval2015/task14/
"""

from pathlib import Path
from typing import Any, Dict, List, Literal

from .base import BaseTaskDataset, TaskRegistry, UnifiedSample


@TaskRegistry.register("semeval2015t14")
class SemEval2015T14Dataset(BaseTaskDataset):
    """SemEval-2015 Task 14 dataset with discontiguous entity support.

    Key difference from 2014: Entities can have multiple non-contiguous spans.

    File structure:
        data/semeval2015t14/
        ├── train/
        │   ├── *.txt
        │   └── *.ann
        ├── dev/
        └── test/

    Annotation format (.ann):
        T1	Disorder 10 15;20 25	chest ... pain
            ^-- Multiple spans separated by semicolon

    IMPORTANT: Must use span-based representation, not BIO tags,
               since BIO cannot represent discontiguous entities.
    """

    LABEL_SCHEMA = {
        "Disorder": 0,
        "Negation": 1,  # May include negation attributes
        "Uncertainty": 2,
    }

    def __init__(self, data_path: Path, split: Literal["train", "dev", "test"]):
        """Initialize dataset.

        Args:
            data_path: Path to semeval2015t14/ directory
            split: Dataset split to load
        """
        super().__init__(
            data_path=data_path,
            split=split,
            task_name="semeval2015t14",
            task_type="span",
            task_level="span",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Parse BRAT annotation files with discontiguous span support.

        Returns:
            List of dictionaries with keys:
                - text: Raw clinical text
                - entities: List of entity dicts, each containing:
                    - spans: List of (start, end) tuples (may be multiple)
                    - label: Entity type
                    - mention: Full text (may be non-contiguous)
                - doc_id: Document identifier
        """
        # TODO: Implement discontiguous span parser
        #
        # Implementation steps:
        # 1. Parse .ann files line by line
        # 2. For each entity annotation (starts with "T"):
        #    - Split span field by ";" to get multiple ranges
        #    - Parse each range as (start, end)
        #    - Store as list of spans per entity
        # 3. Handle attributes (negation, uncertainty) marked with "A" lines
        #
        # Example:
        #   T1	Disorder 10 15;20 25	chest pain
        #   A1	Negation T1
        #
        # Parsed as:
        #   {
        #       "entity_id": "T1",
        #       "spans": [(10, 15), (20, 25)],
        #       "label": "Disorder",
        #       "mention": "chest pain",
        #       "attributes": ["Negation"]
        #   }

        raise NotImplementedError(
            "Discontiguous span parser requires actual data files. "
            "Download SemEval-2015 Task 14 dataset from organizers."
        )

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert discontiguous span annotations to UnifiedSample.

        Args:
            raw_item: Dictionary from parse() with text and entities

        Returns:
            UnifiedSample with span labels (list of (start, end, label) tuples)
        """
        text = raw_item["text"]
        entities = raw_item["entities"]

        # TODO: Convert to span format
        #
        # For each entity with multiple spans, we need to decide:
        # Option 1: Treat each span separately (easier for model)
        # Option 2: Group spans by entity (preserves discontiguous structure)
        #
        # Recommend Option 1 for simplicity:
        #   spans = []
        #   for entity in entities:
        #       for start, end in entity["spans"]:
        #           spans.append((start, end, entity["label"]))

        return UnifiedSample(
            task="semeval2015t14",
            task_type="span",
            task_level="span",
            input_text=text,
            labels=[],  # TODO: List of (start, end, label) tuples
            metadata={
                "doc_id": raw_item.get("doc_id"),
                "has_discontiguous": any(len(e["spans"]) > 1 for e in entities),
            },
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return span label schema."""
        return self.LABEL_SCHEMA
