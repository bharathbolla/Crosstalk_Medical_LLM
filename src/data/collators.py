"""Task-specific collators with token counting.

Each collator:
1. Tokenizes inputs
2. Pads sequences to batch max length
3. Counts tokens (for TokenTracker)
4. Returns task-appropriate batch dictionary
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizer

from .base import UnifiedSample


@dataclass
class CollatedBatch:
    """Standard batch output from all collators.

    This ensures consistent interface for multi-task training loops.
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor  # Format varies by task
    task: str
    token_count: int
    metadata: Optional[Dict] = None


class NERCollator:
    """Collator for token-level sequence labeling (NER, POS, etc.).

    Used by:
    - SemEval-2014 Task 7 (clinical disorder spans)
    - SemEval-2021 Task 6 Level 1 (medication mentions)

    Label format: BIO tags aligned with wordpiece tokens
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        label_to_id: Dict[str, int],
        max_length: int = 512,
        padding: str = "longest",
        ignore_index: int = -100,
    ):
        """Initialize NER collator.

        Args:
            tokenizer: Pretrained tokenizer
            label_to_id: Mapping from BIO tags to label indices
            max_length: Maximum sequence length
            padding: Padding strategy ("longest" or "max_length")
            ignore_index: Label index for padding tokens (typically -100)
        """
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.padding = padding
        self.ignore_index = ignore_index

    def __call__(self, task: str, samples: List[UnifiedSample]) -> CollatedBatch:
        """Collate a batch of NER samples.

        Args:
            task: Task name
            samples: List of UnifiedSample objects

        Returns:
            CollatedBatch with aligned labels
        """
        texts = [s.input_text for s in samples]
        label_sequences = [s.labels for s in samples]  # List of BIO tag lists

        # Tokenize with offset mapping for label alignment
        tokenized = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
            is_split_into_words=False,  # Raw text input
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        offset_mapping = tokenized["offset_mapping"]

        # Align BIO tags with wordpiece tokens
        batch_labels = []
        for i, (offsets, bio_tags) in enumerate(zip(offset_mapping, label_sequences)):
            labels = []
            word_idx = 0

            for j, (start, end) in enumerate(offsets):
                # Special tokens get ignore_index
                if start == 0 and end == 0:
                    labels.append(self.ignore_index)
                else:
                    # Assign BIO tag to token
                    if word_idx < len(bio_tags):
                        tag = bio_tags[word_idx]
                        labels.append(self.label_to_id.get(tag, self.ignore_index))
                        word_idx += 1
                    else:
                        labels.append(self.ignore_index)

            batch_labels.append(labels)

        labels = torch.tensor(batch_labels, dtype=torch.long)

        # Count total tokens (excluding padding)
        token_count = attention_mask.sum().item()

        return CollatedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            task=task,
            token_count=token_count,
        )


class SpanCollator:
    """Collator for span-level classification with discontiguous entities.

    Used by:
    - SemEval-2015 Task 14 (disorder identification with discontinuous spans)

    This task requires identifying start/end positions of entity spans,
    which may be non-contiguous.

    Label format: List of (start, end, label) tuples per sample
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        label_to_id: Dict[str, int],
        max_length: int = 512,
        max_spans: int = 50,
        padding: str = "longest",
    ):
        """Initialize span collator.

        Args:
            tokenizer: Pretrained tokenizer
            label_to_id: Mapping from span labels to indices
            max_length: Maximum sequence length
            max_spans: Maximum number of spans per sample
            padding: Padding strategy
        """
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.max_spans = max_spans
        self.padding = padding

    def __call__(self, task: str, samples: List[UnifiedSample]) -> CollatedBatch:
        """Collate a batch of span samples.

        Args:
            task: Task name
            samples: List of UnifiedSample objects with span labels

        Returns:
            CollatedBatch with span start/end positions and labels
        """
        texts = [s.input_text for s in samples]
        span_lists = [s.labels for s in samples]  # List of [(start, end, label), ...]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        offset_mapping = tokenized["offset_mapping"]

        # Convert character-level spans to token-level spans
        batch_size = len(samples)
        span_starts = torch.zeros((batch_size, self.max_spans), dtype=torch.long)
        span_ends = torch.zeros((batch_size, self.max_spans), dtype=torch.long)
        span_labels = torch.zeros((batch_size, self.max_spans), dtype=torch.long)
        span_mask = torch.zeros((batch_size, self.max_spans), dtype=torch.bool)

        for i, (offsets, spans) in enumerate(zip(offset_mapping, span_lists)):
            for j, (char_start, char_end, label) in enumerate(spans[:self.max_spans]):
                # Find token indices that overlap with character span
                token_start = None
                token_end = None

                for tok_idx, (tok_start, tok_end) in enumerate(offsets):
                    if tok_start <= char_start < tok_end and token_start is None:
                        token_start = tok_idx
                    if tok_start < char_end <= tok_end:
                        token_end = tok_idx
                        break

                if token_start is not None and token_end is not None:
                    span_starts[i, j] = token_start
                    span_ends[i, j] = token_end
                    span_labels[i, j] = self.label_to_id.get(label, 0)
                    span_mask[i, j] = True

        # Stack span features into labels tensor
        # Format: [batch_size, max_spans, 4] where last dim is [start, end, label, mask]
        labels = torch.stack([
            span_starts,
            span_ends,
            span_labels,
            span_mask.long(),
        ], dim=-1)

        token_count = attention_mask.sum().item()

        return CollatedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            task=task,
            token_count=token_count,
        )


class RECollator:
    """Collator for relation extraction between entity pairs.

    Used by:
    - SemEval-2016 Task 12 (temporal relations: EVENT-TIMEX, EVENT-EVENT)
    - SemEval-2021 Task 6 Level 2 (medication-adverse event relations)

    Label format: List of (head_start, head_end, tail_start, tail_end, relation) tuples
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        relation_to_id: Dict[str, int],
        max_length: int = 512,
        max_pairs: int = 30,
        padding: str = "longest",
        add_entity_markers: bool = True,
    ):
        """Initialize RE collator.

        Args:
            tokenizer: Pretrained tokenizer
            relation_to_id: Mapping from relation labels to indices
            max_length: Maximum sequence length
            max_pairs: Maximum number of entity pairs per sample
            padding: Padding strategy
            add_entity_markers: Whether to add [E1], [/E1], [E2], [/E2] markers
        """
        self.tokenizer = tokenizer
        self.relation_to_id = relation_to_id
        self.max_length = max_length
        self.max_pairs = max_pairs
        self.padding = padding
        self.add_entity_markers = add_entity_markers

        # Add special entity markers to tokenizer if needed
        if add_entity_markers:
            special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    def __call__(self, task: str, samples: List[UnifiedSample]) -> CollatedBatch:
        """Collate a batch of RE samples.

        Args:
            task: Task name
            samples: List of UnifiedSample objects with entity pair relations

        Returns:
            CollatedBatch with entity pair positions and relation labels
        """
        texts = [s.input_text for s in samples]
        relation_lists = [s.labels for s in samples]

        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Prepare entity pair tensors
        batch_size = len(samples)
        head_starts = torch.zeros((batch_size, self.max_pairs), dtype=torch.long)
        head_ends = torch.zeros((batch_size, self.max_pairs), dtype=torch.long)
        tail_starts = torch.zeros((batch_size, self.max_pairs), dtype=torch.long)
        tail_ends = torch.zeros((batch_size, self.max_pairs), dtype=torch.long)
        relation_labels = torch.zeros((batch_size, self.max_pairs), dtype=torch.long)
        pair_mask = torch.zeros((batch_size, self.max_pairs), dtype=torch.bool)

        for i, relations in enumerate(relation_lists):
            for j, (h_start, h_end, t_start, t_end, relation) in enumerate(relations[:self.max_pairs]):
                head_starts[i, j] = h_start
                head_ends[i, j] = h_end
                tail_starts[i, j] = t_start
                tail_ends[i, j] = t_end
                relation_labels[i, j] = self.relation_to_id.get(relation, 0)
                pair_mask[i, j] = True

        # Stack into labels tensor
        # Format: [batch_size, max_pairs, 6] where last dim is [h_start, h_end, t_start, t_end, rel, mask]
        labels = torch.stack([
            head_starts,
            head_ends,
            tail_starts,
            tail_ends,
            relation_labels,
            pair_mask.long(),
        ], dim=-1)

        token_count = attention_mask.sum().item()

        return CollatedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            task=task,
            token_count=token_count,
        )


class QACollator:
    """Collator for question-answer pair ranking.

    Used by:
    - SemEval-2017 Task 3 (community question answering)

    Label format: Relevance score (0=irrelevant, 1=relevant) or ranking score
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        padding: str = "longest",
        qa_format: str = "concatenated",  # or "separated"
    ):
        """Initialize QA collator.

        Args:
            tokenizer: Pretrained tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy
            qa_format: How to combine question and answer
                      "concatenated": [CLS] Q [SEP] A [SEP]
                      "separated": Process Q and A separately
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.qa_format = qa_format

    def __call__(self, task: str, samples: List[UnifiedSample]) -> CollatedBatch:
        """Collate a batch of QA samples.

        Args:
            task: Task name
            samples: List of UnifiedSample objects with QA pairs

        Returns:
            CollatedBatch with relevance labels
        """
        # Extract questions, answers, and relevance labels
        questions = [s.metadata.get("question", "") for s in samples]
        answers = [s.input_text for s in samples]  # Answer is the main text
        labels = [s.labels for s in samples]  # Relevance scores

        # Tokenize based on format
        if self.qa_format == "concatenated":
            tokenized = self.tokenizer(
                questions,
                answers,
                padding=self.padding,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:  # separated
            # Tokenize questions and answers separately (for dual-encoder)
            tokenized = self.tokenizer(
                questions,
                padding=self.padding,
                truncation=True,
                max_length=self.max_length // 2,
                return_tensors="pt",
            )
            # Note: For separated format, we'd need to return both Q and A encodings
            # This is a simplified version

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Convert labels to tensor
        label_tensor = torch.tensor(labels, dtype=torch.long)

        token_count = attention_mask.sum().item()

        return CollatedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label_tensor,
            task=task,
            token_count=token_count,
        )
