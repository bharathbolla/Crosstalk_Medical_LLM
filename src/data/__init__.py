"""Data loading and processing for medical NLP tasks."""

from .base import UnifiedSample, BaseTaskDataset, TaskRegistry
from .multitask_loader import MultiTaskBatchSampler, TokenTracker, create_multitask_dataloader
from .collators import NERCollator, SpanCollator, RECollator, QACollator

# Import all task parsers to register them
from .bc2gm import BC2GMDataset
from .jnlpba import JNLPBADataset
from .chemprot import ChemProtDataset  # Re-enabled: Dataset now included in repo
from .ddi import DDIDataset
from .gad import GADDataset
from .hoc import HoCDataset
from .pubmedqa import PubMedQADataset
from .biosses import BIOSSESDataset

__all__ = [
    "UnifiedSample",
    "BaseTaskDataset",
    "TaskRegistry",
    "MultiTaskBatchSampler",
    "TokenTracker",
    "create_multitask_dataloader",
    "NERCollator",
    "SpanCollator",
    "RECollator",
    "QACollator",
    # Task datasets (8 active - all included in repo!)
    "BC2GMDataset",
    "JNLPBADataset",
    "ChemProtDataset",  # Re-enabled: Dataset now in repo
    "DDIDataset",
    "GADDataset",
    "HoCDataset",
    "PubMedQADataset",
    "BIOSSESDataset",
]
