"""Base classes for unified task representation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
from pathlib import Path

import torch
from torch.utils.data import Dataset


TaskType = Literal["ner", "span", "re", "qa", "classification"]
TaskLevel = Literal["token", "span", "pair", "sequence"]


@dataclass
class UnifiedSample:
    """Unified representation for all task types.

    This dataclass ensures consistent structure across all 5 SemEval tasks,
    enabling unified batch sampling and token counting.

    Attributes:
        task: Task identifier (e.g., "semeval2014t7")
        task_type: High-level task category
        task_level: Granularity of prediction
        input_text: Raw input text
        labels: Task-specific labels (BIO tags, spans, relations, etc.)
        metadata: Additional task-specific information
        token_count: Number of tokens in input (after tokenization)
    """
    task: str
    task_type: TaskType
    task_level: TaskLevel
    input_text: str
    labels: Any  # Type varies by task: List[str] for NER, List[tuple] for RE, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    def __post_init__(self):
        """Validate task type and level compatibility."""
        valid_combinations = {
            ("ner", "token"),
            ("span", "span"),
            ("re", "pair"),
            ("qa", "sequence"),
            ("classification", "sequence"),
        }
        if (self.task_type, self.task_level) not in valid_combinations:
            raise ValueError(
                f"Invalid task_type={self.task_type}, task_level={self.task_level} combination"
            )


class TaskRegistry:
    """Global registry for task datasets.

    Enables dynamic task loading and prevents circular imports.
    Used by MultiTaskBatchSampler to fetch datasets by name.
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, task_name: str):
        """Decorator to register a task dataset class.

        Example:
            @TaskRegistry.register("semeval2014t7")
            class SemEval2014T7Dataset(BaseTaskDataset):
                ...
        """
        def decorator(dataset_class):
            cls._registry[task_name] = dataset_class
            return dataset_class
        return decorator

    @classmethod
    def get(cls, task_name: str) -> type:
        """Retrieve a registered task dataset class."""
        if task_name not in cls._registry:
            raise KeyError(
                f"Task '{task_name}' not found. Registered tasks: {list(cls._registry.keys())}"
            )
        return cls._registry[task_name]

    @classmethod
    def list_tasks(cls) -> List[str]:
        """List all registered task names."""
        return list(cls._registry.keys())


class BaseTaskDataset(Dataset, ABC):
    """Abstract base class for all task-specific datasets.

    All task parsers (SemEval2014T7, SemEval2015T14, etc.) must inherit from this
    and implement the abstract methods.

    Design principles:
    - parse() handles raw file I/O and format-specific logic
    - to_unified() converts task-specific format to UnifiedSample
    - Token counting happens during collation (after tokenization)
    """

    def __init__(
        self,
        data_path: Path,
        split: Literal["train", "dev", "test"],
        task_name: str,
        task_type: TaskType,
        task_level: TaskLevel,
    ):
        """Initialize base dataset.

        Args:
            data_path: Path to task data directory
            split: Dataset split to load
            task_name: Unique task identifier
            task_type: Task category (ner, span, re, qa, classification)
            task_level: Prediction granularity
        """
        self.data_path = Path(data_path)
        self.split = split
        self.task_name = task_name
        self.task_type = task_type
        self.task_level = task_level

        # Will be populated by parse()
        self.samples: List[UnifiedSample] = []

        # Parse data on initialization
        self._load_data()

    def _load_data(self):
        """Load and parse data. Called during __init__."""
        raw_data = self.parse()
        self.samples = [self.to_unified(item) for item in raw_data]

    @abstractmethod
    def parse(self) -> List[Dict[str, Any]]:
        """Parse task-specific raw data files.

        Returns:
            List of dictionaries containing task-specific fields.
            Format is task-dependent and will be converted to UnifiedSample.
        """
        pass

    @abstractmethod
    def to_unified(self, raw_item: Dict[str, Any]) -> UnifiedSample:
        """Convert task-specific format to UnifiedSample.

        Args:
            raw_item: Output from parse() for a single example

        Returns:
            UnifiedSample with standardized structure
        """
        pass

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> UnifiedSample:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            UnifiedSample at the given index
        """
        return self.samples[idx]

    def get_label_schema(self) -> Dict[str, int]:
        """Return label-to-index mapping for this task.

        Used by task heads to determine output dimensions.
        Must be implemented by subclasses that have categorical labels.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_label_schema() "
            "or override this method to return None for non-categorical tasks"
        )
