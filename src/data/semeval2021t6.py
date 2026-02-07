"""SemEval-2021 Task 6: Medication Event Classification.

Task: Multi-level annotation of medication-related text
Format: BRAT annotation files
Subtasks:
  Level 1: Medication mention detection (NER)
  Level 2: Relation extraction (medication → adverse events)

We implement both levels in a single dataset (joint NER+RE).

Data source: https://competitions.codalab.org/competitions/26319
"""

from pathlib import Path
from typing import Any, Dict, List, Literal

from .base import BaseTaskDataset, TaskRegistry, UnifiedSample


@TaskRegistry.register("semeval2021t6_level1")
class SemEval2021T6Level1Dataset(BaseTaskDataset):
    """SemEval-2021 Task 6 Level 1: Medication mention detection.

    Entity types:
        - Drug
        - MedicalCondition
        - Symptom
        - DiseaseOrSyndrome
    """

    LABEL_SCHEMA = {
        "O": 0,
        "B-Drug": 1,
        "I-Drug": 2,
        "B-MedicalCondition": 3,
        "I-MedicalCondition": 4,
        "B-Symptom": 5,
        "I-Symptom": 6,
        "B-DiseaseOrSyndrome": 7,
        "I-DiseaseOrSyndrome": 8,
    }

    def __init__(self, data_path: Path, split: Literal["train", "dev", "test"]):
        super().__init__(
            data_path=data_path,
            split=split,
            task_name="semeval2021t6_level1",
            task_type="ner",
            task_level="token",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Parse BRAT files for medication entity detection."""
        # TODO: Implement BRAT parser (similar to 2014 Task 7)
        # Differences:
        # - Multiple entity types (not just Disorder)
        # - May include normalization attributes

        raise NotImplementedError(
            "BRAT parser for SemEval-2021 Task 6 requires actual data files. "
            "Download from CodaLab competition page."
        )

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert to BIO-tagged UnifiedSample."""
        # TODO: Similar to semeval2014t7, but with 4 entity types
        return UnifiedSample(
            task="semeval2021t6_level1",
            task_type="ner",
            task_level="token",
            input_text=raw_item["text"],
            labels=[],  # TODO: BIO tags
            metadata={"doc_id": raw_item.get("doc_id")},
        )

    def get_label_schema(self) -> Dict[str, int]:
        return self.LABEL_SCHEMA


@TaskRegistry.register("semeval2021t6_level2")
class SemEval2021T6Level2Dataset(BaseTaskDataset):
    """SemEval-2021 Task 6 Level 2: Medication-adverse event relations.

    Relation types:
        - Causes (medication → symptom/condition)
        - Treats (medication → disease)
        - Prevents (medication → disease)
    """

    RELATION_SCHEMA = {
        "Causes": 0,
        "Treats": 1,
        "Prevents": 2,
        "NONE": 3,
    }

    def __init__(self, data_path: Path, split: Literal["train", "dev", "test"]):
        super().__init__(
            data_path=data_path,
            split=split,
            task_name="semeval2021t6_level2",
            task_type="re",
            task_level="pair",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Parse BRAT files for medication-event relations.

        BRAT relation format:
            R1	Causes Arg1:T1 Arg2:T2
            where T1 is Drug entity, T2 is Symptom entity
        """
        # TODO: Implement relation parser
        # Need to:
        # 1. Parse entity annotations (T lines)
        # 2. Parse relation annotations (R lines)
        # 3. Link relation endpoints to entity offsets

        raise NotImplementedError(
            "Relation parser for SemEval-2021 Task 6 requires actual data files."
        )

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert to relation-labeled UnifiedSample."""
        # TODO: Format as (head_start, head_end, tail_start, tail_end, relation)
        return UnifiedSample(
            task="semeval2021t6_level2",
            task_type="re",
            task_level="pair",
            input_text=raw_item["text"],
            labels=[],  # TODO: Relation tuples
            metadata={
                "doc_id": raw_item.get("doc_id"),
                "num_entities": len(raw_item.get("entities", [])),
            },
        )

    def get_label_schema(self) -> Dict[str, int]:
        return self.RELATION_SCHEMA
