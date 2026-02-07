"""SemEval-2017 Task 3: Community Question Answering.

Task: Question-answer pair ranking in medical forums
Format: XML files with question-answer threads
Subtasks:
  A: Question-comment similarity
  B: Question-question similarity
  C: Question-external comment similarity

We implement Subtask A (answer ranking for a given question).

Data source: https://alt.qcri.org/semeval2017/task3/
"""

from pathlib import Path
from typing import Any, Dict, List, Literal

from .base import BaseTaskDataset, TaskRegistry, UnifiedSample


@TaskRegistry.register("semeval2017t3")
class SemEval2017T3Dataset(BaseTaskDataset):
    """SemEval-2017 Task 3 dataset for medical QA ranking.

    File structure:
        data/semeval2017t3/
        ├── train/
        │   └── *.xml
        ├── dev/
        └── test/

    XML format:
        <Thread>
            <Question id="Q1" text="What causes chest pain?"/>
            <Comment id="C1" question_id="Q1" relevance="Good" text="..."/>
            <Comment id="C2" question_id="Q1" relevance="PotentiallyUseful" text="..."/>
            <Comment id="C3" question_id="Q1" relevance="Bad" text="..."/>
        </Thread>

    Relevance labels:
        Good (2), PotentiallyUseful (1), Bad (0)
    """

    RELEVANCE_SCHEMA = {
        "Bad": 0,
        "PotentiallyUseful": 1,
        "Good": 2,
    }

    def __init__(self, data_path: Path, split: Literal["train", "dev", "test"]):
        """Initialize dataset.

        Args:
            data_path: Path to semeval2017t3/ directory
            split: Dataset split to load
        """
        super().__init__(
            data_path=data_path,
            split=split,
            task_name="semeval2017t3",
            task_type="qa",
            task_level="sequence",
        )

    def parse(self) -> List[Dict[str, Any]]:
        """Parse QA thread XML files.

        Returns:
            List of dictionaries with keys:
                - question_id: Question identifier
                - question_text: Question text
                - answer_id: Answer/comment identifier
                - answer_text: Answer text
                - relevance: Relevance label (Good/PotentiallyUseful/Bad)
                - metadata: Additional thread info
        """
        # TODO: Implement QA XML parser
        #
        # Implementation steps:
        # 1. Parse XML files to extract <Question> and <Comment> elements
        # 2. For each comment, pair it with its corresponding question
        # 3. Extract relevance label from comment attributes
        # 4. Flatten into question-answer pairs
        #
        # Example:
        #   import xml.etree.ElementTree as ET
        #   tree = ET.parse(xml_file)
        #   root = tree.getroot()
        #
        #   qa_pairs = []
        #   for thread in root.findall('.//Thread'):
        #       question = thread.find('Question')
        #       q_id = question.get('id')
        #       q_text = question.get('text')
        #
        #       for comment in thread.findall('Comment'):
        #           if comment.get('question_id') == q_id:
        #               qa_pairs.append({
        #                   'question_id': q_id,
        #                   'question_text': q_text,
        #                   'answer_id': comment.get('id'),
        #                   'answer_text': comment.get('text'),
        #                   'relevance': comment.get('relevance'),
        #               })

        raise NotImplementedError(
            "QA XML parser requires actual data files. "
            "Download SemEval-2017 Task 3 dataset from organizers."
        )

    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert QA pair to UnifiedSample.

        Args:
            raw_item: Dictionary from parse() with question, answer, relevance

        Returns:
            UnifiedSample with relevance label
        """
        question_text = raw_item["question_text"]
        answer_text = raw_item["answer_text"]
        relevance = raw_item["relevance"]

        # Store question in metadata, answer as input_text
        return UnifiedSample(
            task="semeval2017t3",
            task_type="qa",
            task_level="sequence",
            input_text=answer_text,
            labels=self.RELEVANCE_SCHEMA[relevance],  # Single integer label
            metadata={
                "question": question_text,
                "question_id": raw_item["question_id"],
                "answer_id": raw_item["answer_id"],
            },
        )

    def get_label_schema(self) -> Dict[str, int]:
        """Return relevance label schema."""
        return self.RELEVANCE_SCHEMA
