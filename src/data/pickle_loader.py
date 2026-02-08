"""
Pickle-based dataset loader - works without datasets library!
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any
from .base import UnifiedSample, BaseTaskDataset


class PickleDataset(BaseTaskDataset):
    """Base class for loading datasets from pickle files."""

    def __init__(self, data_path: Path, split: str = "train", **kwargs):
        """
        Initialize dataset from pickle file.

        Args:
            data_path: Path to data directory
            split: Dataset split (train/validation/test)
        """
        self.data_path = Path(data_path)
        self.split = split

        # Load pickle file
        pickle_path = self.data_path.parent / "pickle" / f"{self.task_name}.pkl"

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"Pickle file not found: {pickle_path}\n"
                f"Run convert_to_simple_format.py locally to create pickle files."
            )

        with open(pickle_path, 'rb') as f:
            all_data = pickle.load(f)

        if split not in all_data:
            raise ValueError(
                f"Split '{split}' not found in {self.task_name}. "
                f"Available splits: {list(all_data.keys())}"
            )

        self.raw_data = all_data[split]
        self.samples = self._parse_samples()

    def _parse_samples(self) -> List[UnifiedSample]:
        """
        Parse raw data into UnifiedSamples.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _parse_samples()")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> UnifiedSample:
        return self.samples[idx]


def check_pickle_available() -> bool:
    """Check if pickle files are available."""
    from pathlib import Path
    pickle_dir = Path("data/pickle")
    return pickle_dir.exists() and any(pickle_dir.glob("*.pkl"))
