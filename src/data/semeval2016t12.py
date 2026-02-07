"""SemEval-2016 Task 12: Clinical TempEval.

Task: Temporal relation extraction in clinical notes
Format: XML files with EVENT, TIMEX3, and TLINK annotations
Subtasks:
  TS: TIMEX3 span detection
  ES: EVENT span detection
  TA: TIMEX3 attribute classification
  EA: EVENT attribute classification
  DR: Document-time relation classification (EVENT-DCT)
  CR: Narrative container relation (EVENT-EVENT, EVENT-TIMEX)

We implement DR and CR (relation extraction tasks).

Data source: https://alt.qcri.org/semeval2016/task12/
"""

from pathlib import Path
from typing import Any, Dict, List, Literal

from .base import BaseTaskDataset, TaskRegistry, UnifiedSample


@TaskRegistry.register("semeval2016t12")
class SemEval2016T12Dataset(BaseTaskDataset):
    """SemEval-2016 Task 12 dataset for temporal relation extraction.

    File structure:
        data/semeval2016t12/
        ├── train/
        │   └── *.xml (TimeML format)
        ├── dev/
        └── test/

    XML format:
        <TEXT>Patient had <EVENT id="E1">surgery</EVENT> on
        <TIMEX3 id="T1">2015-01-15</TIMEX3>.</TEXT>
        <TLINK eventID="E1" relatedToTime="T1" relType="OVERLAP"/>

    Relation types (simplified):
        BEFORE, AFTER, OVERLAP, BEFORE-OVERLAP, CONTAINS
    """

    RELATION_SCHEMA = {
        "BEFORE": 0,
        "AFTER": 1,
        "OVERLAP": 2,
        "BEFORE-OVERLAP": 3,
        "CONTAINS": 4,
        "NONE": 5,  # No relation
    }

    def __init__(self, data_path: Path, split: Literal["train", "dev", "test"]):
        """Initialize dataset.

        Args:
            data_path: Path to semeval2016t12/ directory
            split: Dataset split to load
        """
        super().__init__(
            data_path=data_path,
            split=split,
            task_name="semeval2016t12",
            task_type="re",
            task_level="pair",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Parse TimeML XML files.

        Returns:
            List of dictionaries with keys:
                - text: Raw clinical text (without XML tags)
                - events: List of (id, start, end, text, attributes) tuples
                - timexes: List of (id, start, end, text, attributes) tuples
                - relations: List of (head_id, tail_id, relation_type) tuples
                - doc_id: Document identifier
        """
        # TODO: Implement TimeML XML parser
        #
        # Implementation steps:
        # 1. Use xml.etree.ElementTree to parse XML
        # 2. Extract <TEXT> content and strip tags to get raw text
        # 3. Parse <EVENT> tags to get event mentions and attributes
        # 4. Parse <TIMEX3> tags to get temporal expressions
        # 5. Parse <TLINK> tags to get temporal relations
        # 6. Convert character offsets after stripping tags
        #
        # Example:
        #   import xml.etree.ElementTree as ET
        #   tree = ET.parse(xml_file)
        #   root = tree.getroot()
        #   text_elem = root.find('.//TEXT')
        #   # Strip tags but track offsets...
        #   tlinks = root.findall('.//TLINK')
        #   for tlink in tlinks:
        #       head = tlink.get('eventID') or tlink.get('timeID')
        #       tail = tlink.get('relatedToEvent') or tlink.get('relatedToTime')
        #       rel_type = tlink.get('relType')

        raise NotImplementedError(
            "TimeML parser requires actual data files. "
            "Download SemEval-2016 Task 12 dataset from organizers."
        )

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert temporal relations to UnifiedSample.

        Args:
            raw_item: Dictionary from parse() with events, timexes, relations

        Returns:
            UnifiedSample with relation labels
        """
        text = raw_item["text"]
        events = raw_item["events"]
        timexes = raw_item["timexes"]
        relations = raw_item["relations"]

        # TODO: Convert to relation format
        #
        # Format: [(head_start, head_end, tail_start, tail_end, relation_type), ...]
        #
        # Need to map entity IDs to character offsets:
        #   entity_offsets = {}
        #   for id, start, end, _, _ in events + timexes:
        #       entity_offsets[id] = (start, end)
        #
        #   relation_tuples = []
        #   for head_id, tail_id, rel_type in relations:
        #       h_start, h_end = entity_offsets[head_id]
        #       t_start, t_end = entity_offsets[tail_id]
        #       relation_tuples.append((h_start, h_end, t_start, t_end, rel_type))

        return UnifiedSample(
            task="semeval2016t12",
            task_type="re",
            task_level="pair",
            input_text=text,
            labels=[],  # TODO: List of (h_start, h_end, t_start, t_end, relation) tuples
            metadata={
                "doc_id": raw_item.get("doc_id"),
                "num_events": len(events),
                "num_timexes": len(timexes),
            },
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return temporal relation schema."""
        return self.RELATION_SCHEMA
