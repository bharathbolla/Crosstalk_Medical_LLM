"""Task-specific prediction heads for multi-task learning.

Each head is designed for a specific task type:
- TokenClassificationHead: BIO tagging (Tasks 2014-T7, 2021-T6 NER)
- SpanClassificationHead: Discontiguous entity detection (Task 2015-T14)
- RelationExtractionHead: Entity pair classification (Tasks 2016-T12, 2021-T6 RE)
- SequenceRankingHead: QA pair ranking (Task 2017-T3)
"""

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenClassificationHead(nn.Module):
    """Standard token-level classification head for NER (BIO/BIOES tagging).

    Used by:
        - SemEval-2014 Task 7 (disorder span detection)
        - SemEval-2021 Task 6 Level 1 (medication mention detection)
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
    ):
        """Initialize token classification head.

        Args:
            hidden_size: Hidden dimension from encoder
            num_labels: Number of BIO labels
            dropout: Dropout probability
        """
        super().__init__()
        self.num_labels = num_labels

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for token classification.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            labels: [batch_size, seq_len] (optional, for loss computation)

        Returns:
            logits: [batch_size, seq_len, num_labels]
            loss: Scalar loss (if labels provided)
        """
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return logits, loss


class SpanClassificationHead(nn.Module):
    """Span-level classification head for discontiguous entities.

    Used by:
        - SemEval-2015 Task 14 (discontiguous disorder spans)

    Architecture:
        1. Enumerate all candidate spans (up to max_span_len)
        2. Create span representations: [start; end; width_embedding]
        3. Classify each span (entity type or null)
        4. For non-null spans, compute pairwise linking scores
        5. Group linked spans into entities
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        max_span_len: int = 10,
        dropout: float = 0.1,
        use_span_width_embedding: bool = True,
    ):
        """Initialize span classification head.

        Args:
            hidden_size: Hidden dimension
            num_labels: Number of entity types (+ null)
            max_span_len: Maximum span length to consider
            dropout: Dropout probability
            use_span_width_embedding: Whether to use width embeddings
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.max_span_len = max_span_len
        self.use_span_width_embedding = use_span_width_embedding

        # Span width embeddings
        if use_span_width_embedding:
            self.width_embedding = nn.Embedding(max_span_len, hidden_size // 4)
            span_repr_size = hidden_size * 2 + hidden_size // 4
        else:
            span_repr_size = hidden_size * 2

        # Span classifier
        self.dropout = nn.Dropout(dropout)
        self.span_classifier = nn.Sequential(
            nn.Linear(span_repr_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

        # Span linking scorer (for grouping discontiguous spans)
        self.link_scorer = nn.Bilinear(span_repr_size, span_repr_size, 1)

    def _enumerate_spans(
        self,
        seq_len: int,
        max_span_len: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Enumerate all candidate spans.

        Args:
            seq_len: Sequence length
            max_span_len: Maximum span length (default: self.max_span_len)

        Returns:
            List of (start, end) tuples
        """
        if max_span_len is None:
            max_span_len = self.max_span_len

        spans = []
        for start in range(seq_len):
            for end in range(start, min(start + max_span_len, seq_len)):
                spans.append((start, end))

        return spans

    def _create_span_representation(
        self,
        hidden_states: torch.Tensor,
        start: int,
        end: int,
    ) -> torch.Tensor:
        """Create span representation from start and end hidden states.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            start: Start index
            end: End index

        Returns:
            Span representation: [batch_size, span_repr_size]
        """
        start_repr = hidden_states[:, start, :]
        end_repr = hidden_states[:, end, :]

        if self.use_span_width_embedding:
            width = end - start
            width_embed = self.width_embedding(
                torch.tensor(width, device=hidden_states.device)
            )
            # Expand to batch size
            width_embed = width_embed.unsqueeze(0).expand(hidden_states.size(0), -1)
            span_repr = torch.cat([start_repr, end_repr, width_embed], dim=-1)
        else:
            span_repr = torch.cat([start_repr, end_repr], dim=-1)

        return span_repr

    def forward(
        self,
        hidden_states: torch.Tensor,
        span_labels: Optional[torch.Tensor] = None,
        link_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass for span classification.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            span_labels: [batch_size, num_spans] (optional)
            link_labels: [batch_size, num_spans, num_spans] (optional)

        Returns:
            span_logits: [batch_size, num_spans, num_labels]
            link_scores: [batch_size, num_spans, num_spans]
            loss: Combined loss (if labels provided)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Enumerate all candidate spans
        candidate_spans = self._enumerate_spans(seq_len)
        num_spans = len(candidate_spans)

        # Create span representations
        span_reprs = []
        for start, end in candidate_spans:
            span_repr = self._create_span_representation(hidden_states, start, end)
            span_reprs.append(span_repr)

        span_reprs = torch.stack(span_reprs, dim=1)  # [batch_size, num_spans, span_repr_size]

        # Classify spans
        span_reprs_dropout = self.dropout(span_reprs)
        span_logits = self.span_classifier(span_reprs_dropout)  # [batch_size, num_spans, num_labels]

        # Compute pairwise linking scores
        # For each pair of spans, predict if they belong to the same entity
        link_scores = torch.zeros(batch_size, num_spans, num_spans, device=hidden_states.device)
        for i in range(num_spans):
            for j in range(i + 1, num_spans):
                score = self.link_scorer(span_reprs[:, i], span_reprs[:, j]).squeeze(-1)
                link_scores[:, i, j] = score
                link_scores[:, j, i] = score  # Symmetric

        # Compute loss if labels provided
        loss = None
        if span_labels is not None:
            # Span classification loss
            span_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            span_loss = span_loss_fct(
                span_logits.view(-1, self.num_labels),
                span_labels.view(-1)
            )

            # Span linking loss (if link labels provided)
            if link_labels is not None:
                link_loss_fct = nn.BCEWithLogitsLoss()
                link_loss = link_loss_fct(link_scores.view(-1), link_labels.view(-1))
                loss = span_loss + link_loss
            else:
                loss = span_loss

        return span_logits, link_scores, loss


class RelationExtractionHead(nn.Module):
    """Relation extraction head for entity pair classification.

    Used by:
        - SemEval-2016 Task 12 (temporal relations: EVENT-TIMEX, EVENT-EVENT)
        - SemEval-2021 Task 6 Level 2 (medication-adverse event relations)

    Takes entity span positions and predicts relation types.
    """

    def __init__(
        self,
        hidden_size: int,
        num_relations: int,
        dropout: float = 0.1,
        use_entity_markers: bool = True,
    ):
        """Initialize relation extraction head.

        Args:
            hidden_size: Hidden dimension
            num_relations: Number of relation types (+ NONE)
            dropout: Dropout probability
            use_entity_markers: Whether to use special entity markers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relations = num_relations
        self.use_entity_markers = use_entity_markers

        # Entity pair representation
        # [head_start; head_end; tail_start; tail_end; head*tail]
        pair_repr_size = hidden_size * 5

        # Relation classifier
        self.dropout = nn.Dropout(dropout)
        self.relation_classifier = nn.Sequential(
            nn.Linear(pair_repr_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_relations),
        )

    def _create_entity_representation(
        self,
        hidden_states: torch.Tensor,
        start: torch.Tensor,
        end: torch.Tensor,
    ) -> torch.Tensor:
        """Create entity representation from span.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            start: Start indices [batch_size]
            end: End indices [batch_size]

        Returns:
            Entity representation: [batch_size, hidden_size * 2]
        """
        batch_size = hidden_states.size(0)

        # Gather start and end representations
        start_repr = hidden_states[torch.arange(batch_size), start]
        end_repr = hidden_states[torch.arange(batch_size), end]

        entity_repr = torch.cat([start_repr, end_repr], dim=-1)

        return entity_repr

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_start: torch.Tensor,
        head_end: torch.Tensor,
        tail_start: torch.Tensor,
        tail_end: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for relation extraction.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            head_start: Head entity start indices [batch_size]
            head_end: Head entity end indices [batch_size]
            tail_start: Tail entity start indices [batch_size]
            tail_end: Tail entity end indices [batch_size]
            labels: Relation labels [batch_size] (optional)

        Returns:
            logits: [batch_size, num_relations]
            loss: Scalar loss (if labels provided)
        """
        # Create entity representations
        head_repr = self._create_entity_representation(hidden_states, head_start, head_end)
        tail_repr = self._create_entity_representation(hidden_states, tail_start, tail_end)

        # Create pair representation
        # Concatenate head, tail, and their element-wise product
        head_start_hidden = head_repr[:, :self.hidden_size]
        tail_start_hidden = tail_repr[:, :self.hidden_size]
        interaction = head_start_hidden * tail_start_hidden

        pair_repr = torch.cat([head_repr, tail_repr, interaction], dim=-1)

        # Classify relation
        pair_repr = self.dropout(pair_repr)
        logits = self.relation_classifier(pair_repr)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits, labels)

        return logits, loss


class SequenceRankingHead(nn.Module):
    """Sequence ranking head for QA pair ranking.

    Used by:
        - SemEval-2017 Task 3 (community question answering)

    Takes [CLS] representation of question-answer pair and outputs relevance score.
    Trained with pairwise ranking loss.
    """

    def __init__(
        self,
        hidden_size: int,
        num_relevance_levels: int = 3,  # Bad, PotentiallyUseful, Good
        dropout: float = 0.1,
    ):
        """Initialize sequence ranking head.

        Args:
            hidden_size: Hidden dimension
            num_relevance_levels: Number of relevance levels
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_relevance_levels = num_relevance_levels

        # Relevance scorer
        self.dropout = nn.Dropout(dropout)
        self.relevance_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_relevance_levels),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for sequence ranking.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            labels: Relevance labels [batch_size] (optional)

        Returns:
            logits: [batch_size, num_relevance_levels]
            loss: Scalar loss (if labels provided)
        """
        # Use [CLS] token representation (first token)
        cls_repr = hidden_states[:, 0, :]

        # Score relevance
        cls_repr = self.dropout(cls_repr)
        logits = self.relevance_scorer(cls_repr)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return logits, loss


def create_task_head(
    task_type: str,
    hidden_size: int,
    num_labels: int,
    **kwargs,
) -> nn.Module:
    """Factory function to create task-specific head.

    Args:
        task_type: Type of task ("ner", "span", "re", "qa")
        hidden_size: Hidden dimension
        num_labels: Number of labels/classes
        **kwargs: Additional head-specific arguments

    Returns:
        Task-specific head module
    """
    if task_type == "ner":
        return TokenClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            **kwargs
        )
    elif task_type == "span":
        return SpanClassificationHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            **kwargs
        )
    elif task_type == "re":
        return RelationExtractionHead(
            hidden_size=hidden_size,
            num_relations=num_labels,
            **kwargs
        )
    elif task_type == "qa":
        return SequenceRankingHead(
            hidden_size=hidden_size,
            num_relevance_levels=num_labels,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
