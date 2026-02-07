"""Hierarchical Multi-Task Learning architecture (S3b).

PRIMARY NOVELTY: Two-level hierarchy where low-level tasks (NER) provide
features to high-level tasks (RE, QA, temporal).

Architecture:
    Level 1 (NER): semeval2014t7, semeval2015t14, semeval2021t6_level1
    Level 2 (RE/QA): semeval2016t12, semeval2017t3, semeval2021t6_level2

Level 2 tasks attend to Level 1 features via CrossLevelAttention.
"""

from typing import Dict, List, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class CrossLevelAttention(nn.Module):
    """Cross-attention from Level 2 to Level 1 features.

    Level 2 tasks (RE, QA) query Level 1 task (NER) outputs to extract
    entity-aware representations.

    Key design choice: Should we detach Level 1 gradients?
        - detach=True: L1 is frozen feature extractor for L2 (safer)
        - detach=False: L2 training improves L1 (more capacity, but risk interference)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        detach_l1: bool = False,
    ):
        """Initialize cross-level attention.

        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            detach_l1: Whether to detach Level 1 gradients
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.detach_l1 = detach_l1

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # Cross-attention projections
        # Q from Level 2, K/V from Level 1
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        level2_hidden: torch.Tensor,
        level1_hidden: torch.Tensor,
        level1_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attention from L2 to L1.

        Args:
            level2_hidden: Level 2 features [batch_size, seq_len, hidden_size]
            level1_hidden: Level 1 features [batch_size, seq_len, hidden_size]
            level1_mask: Attention mask for Level 1 [batch_size, seq_len]

        Returns:
            Enhanced Level 2 features [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = level2_hidden.shape

        # Detach Level 1 gradients if specified
        if self.detach_l1:
            level1_hidden = level1_hidden.detach()

        # Project to Q, K, V
        Q = self.q_proj(level2_hidden)  # Query from Level 2
        K = self.k_proj(level1_hidden)  # Key from Level 1
        V = self.v_proj(level1_hidden)  # Value from Level 1

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask if provided
        if level1_mask is not None:
            # Expand mask for multi-head attention
            # level1_mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            mask = level1_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and apply to values
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # Output projection with residual
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = self.layer_norm(output + level2_hidden)

        return output


class HierarchicalAdapter(nn.Module):
    """Hierarchical adapter with two levels.

    Each level has its own LoRA adapter, and Level 2 can attend to Level 1.
    """

    def __init__(
        self,
        base_model: nn.Module,
        shared_rank: int = 16,
        level1_rank: int = 8,
        level2_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        use_cross_attention: bool = True,
        detach_l1_for_l2: bool = False,
    ):
        """Initialize hierarchical adapter.

        Args:
            base_model: Base pretrained model
            shared_rank: Rank for shared LoRA
            level1_rank: Rank for Level 1 adapter
            level2_rank: Rank for Level 2 adapter
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: Target modules for LoRA
            use_cross_attention: Whether to use cross-level attention
            detach_l1_for_l2: Whether to detach L1 gradients when computing L2
        """
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.detach_l1_for_l2 = detach_l1_for_l2

        # Default target modules
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        # Shared adapter (used by all tasks)
        shared_config = LoraConfig(
            r=shared_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.shared_adapter = get_peft_model(base_model, shared_config)

        # Level 1 adapter (NER tasks)
        # Note: In practice, this would be added on top of shared adapter
        self.level1_adapter = nn.Identity()  # Placeholder

        # Level 2 adapter (RE/QA tasks)
        self.level2_adapter = nn.Identity()  # Placeholder

        # Cross-level attention
        if use_cross_attention:
            hidden_size = base_model.config.hidden_size
            self.cross_attention = CrossLevelAttention(
                hidden_size=hidden_size,
                num_heads=8,
                dropout=0.1,
                detach_l1=detach_l1_for_l2,
            )
        else:
            self.cross_attention = None

    def forward_level1(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for Level 1 tasks (NER).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            **kwargs: Additional model arguments

        Returns:
            Level 1 hidden states
        """
        # Shared adapter
        outputs = self.shared_adapter(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        shared_hidden = outputs.hidden_states[-1]

        # Level 1 adapter
        # TODO: Apply level1_adapter on top of shared
        level1_hidden = shared_hidden  # Placeholder

        return level1_hidden

    def forward_level2(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        level1_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for Level 2 tasks (RE, QA).

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            level1_features: Pre-computed Level 1 features (optional)
            **kwargs: Additional model arguments

        Returns:
            Level 2 hidden states
        """
        # Shared adapter
        outputs = self.shared_adapter(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        shared_hidden = outputs.hidden_states[-1]

        # If Level 1 features provided and cross-attention enabled, use them
        if level1_features is not None and self.cross_attention is not None:
            # Cross-attend to Level 1
            enhanced_hidden = self.cross_attention(
                level2_hidden=shared_hidden,
                level1_hidden=level1_features,
                level1_mask=attention_mask,
            )
        else:
            enhanced_hidden = shared_hidden

        # Level 2 adapter
        # TODO: Apply level2_adapter
        level2_hidden = enhanced_hidden  # Placeholder

        return level2_hidden


class HierarchicalMTLModel(nn.Module):
    """Hierarchical Multi-Task Learning Model (S3b).

    PRIMARY ARCHITECTURAL NOVELTY.

    Two-level hierarchy:
        Level 1: NER tasks (semeval2014t7, semeval2015t14, semeval2021t6_l1)
        Level 2: RE/QA tasks (semeval2016t12, semeval2017t3, semeval2021t6_l2)

    Level 2 tasks can optionally attend to Level 1 features, enabling
    entity-aware representations for relation extraction and QA.

    Key ablation flags:
        - use_cross_attention: Enable/disable cross-level attention
        - detach_l1_for_l2: Whether L2 gradients flow back through L1
    """

    # Task hierarchy definition
    LEVEL1_TASKS = [
        "semeval2014t7",
        "semeval2015t14",
        "semeval2021t6_level1",
    ]

    LEVEL2_TASKS = [
        "semeval2016t12",
        "semeval2017t3",
        "semeval2021t6_level2",
    ]

    def __init__(
        self,
        base_model: nn.Module,
        shared_rank: int = 16,
        level1_rank: int = 8,
        level2_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        use_cross_attention: bool = True,
        detach_l1_for_l2: bool = False,
    ):
        """Initialize hierarchical MTL model.

        Args:
            base_model: Base pretrained model
            shared_rank: Rank for shared LoRA (applied to all tasks)
            level1_rank: Rank for Level 1 adapter (NER)
            level2_rank: Rank for Level 2 adapter (RE/QA)
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            target_modules: Target modules for LoRA
            use_cross_attention: Whether to use cross-level attention
            detach_l1_for_l2: Whether to detach L1 gradients for L2
        """
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.detach_l1_for_l2 = detach_l1_for_l2

        # Initialize hierarchical adapter
        self.adapter = HierarchicalAdapter(
            base_model=base_model,
            shared_rank=shared_rank,
            level1_rank=level1_rank,
            level2_rank=level2_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            use_cross_attention=use_cross_attention,
            detach_l1_for_l2=detach_l1_for_l2,
        )

        # Cache for Level 1 features (used by Level 2 if enabled)
        self.level1_cache: Optional[Dict[str, torch.Tensor]] = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        use_cached_l1: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with hierarchical routing.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            task: Task name (determines routing)
            use_cached_l1: Whether to use cached Level 1 features (for Level 2)
            **kwargs: Additional model arguments

        Returns:
            Hidden states for the task
        """
        if task in self.LEVEL1_TASKS:
            # Level 1 task (NER)
            hidden = self.adapter.forward_level1(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )

            # Cache for potential Level 2 use
            if self.level1_cache is None:
                self.level1_cache = {}
            self.level1_cache[task] = hidden.detach() if self.detach_l1_for_l2 else hidden

            return hidden

        elif task in self.LEVEL2_TASKS:
            # Level 2 task (RE/QA)
            # Optionally use cached Level 1 features
            level1_features = None
            if use_cached_l1 and self.level1_cache is not None and len(self.level1_cache) > 0:
                # Use any cached Level 1 feature (or average if multiple)
                level1_features = torch.stack(list(self.level1_cache.values())).mean(dim=0)

            hidden = self.adapter.forward_level2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                level1_features=level1_features,
                **kwargs
            )

            return hidden

        else:
            raise ValueError(
                f"Unknown task: {task}. "
                f"Valid: {self.LEVEL1_TASKS + self.LEVEL2_TASKS}"
            )

    def clear_cache(self):
        """Clear Level 1 feature cache."""
        self.level1_cache = None

    def get_task_level(self, task: str) -> Literal["level1", "level2"]:
        """Get the level for a task.

        Args:
            task: Task name

        Returns:
            "level1" or "level2"
        """
        if task in self.LEVEL1_TASKS:
            return "level1"
        elif task in self.LEVEL2_TASKS:
            return "level2"
        else:
            raise ValueError(f"Unknown task: {task}")

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_params_by_level(self) -> Dict[str, int]:
        """Get parameter counts by level.

        Returns:
            Dictionary with keys: "shared", "level1", "level2", "total"
        """
        shared_params = sum(
            p.numel() for p in self.adapter.shared_adapter.parameters()
            if p.requires_grad
        )
        # TODO: Count level1 and level2 adapter params separately
        level1_params = 0  # Placeholder
        level2_params = 0  # Placeholder

        return {
            "shared": shared_params,
            "level1": level1_params,
            "level2": level2_params,
            "total": shared_params + level1_params + level2_params,
        }
