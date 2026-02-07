"""SemEval-2014 Task 7: Analysis of Clinical Text.

Task: Disorder span detection and normalization
Format: BRAT annotation files (.txt + .ann)
Subtasks:
  A: Disorder mention recognition (NER)
  B: Disorder template slot filling

We implement Subtask A (NER) using BIO tagging scheme.

Data source: https://www.cs.york.ac.uk/semeval-2014/task7/
"""

from pathlib import Path
from typing import Any, Dict, List, Literal

from .base import BaseTaskDataset, TaskRegistry, UnifiedSample


@TaskRegistry.register("semeval2014t7")
class SemEval2014T7Dataset(BaseTaskDataset):
    """SemEval-2014 Task 7 dataset for disorder span detection.

    File structure:
        data/semeval2014t7/
        ├── train/
        │   ├── *.txt (raw text)
        │   └── *.ann (BRAT annotations)
        ├── dev/
        └── test/

    Annotation format (.ann):
        T1	Disorder 45 58	abdominal pain
        T2	Disorder 78 85	nausea

    Output format:
        BIO tags aligned with tokens
    """

    LABEL_SCHEMA = {
        "O": 0,
        "B-Disorder": 1,
        "I-Disorder": 2,
    }

    def __init__(self, data_path: Path, split: Literal["train", "dev", "test"]):
        """Initialize dataset.

        Args:
            data_path: Path to semeval2014t7/ directory
            split: Dataset split to load
        """
        super().__init__(
            data_path=data_path,
            split=split,
            task_name="semeval2014t7",
            task_type="ner",
            task_level="token",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Parse BRAT annotation files.

        Returns:
            List of dictionaries with keys:
                - text: Raw clinical text
                - annotations: List of (start, end, label, mention) tuples
                - doc_id: Document identifier
        """
        # TODO: Implement BRAT parser after PhysioNet access is granted
        #
        # Implementation steps:
        # 1. Find all .txt files in self.data_path / self.split
        # 2. For each .txt file, read raw text
        # 3. Read corresponding .ann file and parse lines starting with "T"
        # 4. Extract start/end character offsets and entity type
        # 5. Return list of {text, annotations, doc_id} dictionaries
        #
        # Example parsing:
        #   with open(txt_file) as f:
        #       text = f.read()
        #   with open(ann_file) as f:
        #       for line in f:
        #           if line.startswith('T'):
        #               parts = line.strip().split('\t')
        #               entity_info = parts[1].split()
        #               entity_type = entity_info[0]
        #               start = int(entity_info[1])
        #               end = int(entity_info[2])
        #               mention = parts[2]
        #               annotations.append((start, end, entity_type, mention))

        raise NotImplementedError(
            "BRAT parser requires actual data files. "
            "Apply for PhysioNet access and download SemEval-2014 Task 7 dataset."
        )

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert BRAT annotations to BIO-tagged UnifiedSample.

        Args:
            raw_item: Dictionary from parse() with text and annotations

        Returns:
            UnifiedSample with BIO tags
        """
        text = raw_item["text"]
        annotations = raw_item["annotations"]  # [(start, end, label, mention), ...]

        # TODO: Convert character-level annotations to BIO tags
        #
        # Implementation steps:
        # 1. Tokenize text (simple whitespace split for now, will be refined in collator)
        # 2. For each token, determine if it overlaps with any annotation
        # 3. Assign B-Disorder to first token of entity, I-Disorder to continuation
        # 4. Assign O to non-entity tokens
        #
        # Example:
        #   tokens = text.split()
        #   bio_tags = ['O'] * len(tokens)
        #   for start, end, label, mention in annotations:
        #       # Find token indices that overlap [start, end)
        #       first_token_idx = ...
        #       last_token_idx = ...
        #       bio_tags[first_token_idx] = f'B-{label}'
        #       for i in range(first_token_idx + 1, last_token_idx + 1):
        #           bio_tags[i] = f'I-{label}'

        # Placeholder return
        return UnifiedSample(
            task="semeval2014t7",
            task_type="ner",
            task_level="token",
            input_text=text,
            labels=[],  # TODO: Replace with BIO tags
            metadata={"doc_id": raw_item.get("doc_id")},
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return BIO label schema."""
        return self.LABEL_SCHEMA
