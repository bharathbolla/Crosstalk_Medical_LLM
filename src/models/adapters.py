"""Shared-Private LoRA adapters with fusion mechanisms.

This module implements the core adapter architecture for multi-task learning:
- SharedPrivateLoRA: Shared adapter (r=16) + per-task private adapters (r=8)
- AttentionFusion: Learns to weight shared vs private outputs
- GatedResidualFusion: Gate-based weighted combination
- Parameter counting utilities for enforcing parameter parity
"""

from typing import Dict, List, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel


class AttentionFusion(nn.Module):
    """Learns to weight shared vs private adapter outputs via attention.

    Computes:
        h_fused = softmax(Q @ K^T) @ [h_shared; h_private]
    where Q = W_q @ h_shared, K = W_k @ [h_shared; h_private]

    This allows the model to dynamically decide how much to rely on
    shared knowledge vs task-specific knowledge.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """Initialize attention fusion module.

        Args:
            hidden_size: Hidden dimension size
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Attention weights
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        shared_output: torch.Tensor,
        private_output: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse shared and private outputs via attention.

        Args:
            shared_output: [batch_size, seq_len, hidden_size]
            private_output: [batch_size, seq_len, hidden_size]

        Returns:
            Fused output: [batch_size, seq_len, hidden_size]
        """
        # Query from shared (what do we need?)
        Q = self.W_q(shared_output)  # [B, L, H]

        # Key and Value from concatenation (what do we have?)
        # Stack shared and private along a new dimension
        stacked = torch.stack([shared_output, private_output], dim=2)  # [B, L, 2, H]

        # Compute attention scores
        # Q: [B, L, H] -> [B, L, 1, H]
        # stacked: [B, L, 2, H] -> [B, L, H, 2] (transpose last two dims)
        Q = Q.unsqueeze(2)  # [B, L, 1, H]
        K = stacked.transpose(-2, -1)  # [B, L, H, 2]

        # Attention scores: [B, L, 1, 2]
        attn_scores = torch.matmul(Q, K) / (self.hidden_size ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L, 1, 2]

        # Apply attention to values
        V = stacked  # [B, L, 2, H]
        fused = torch.matmul(attn_weights, V).squeeze(2)  # [B, L, H]

        # Output projection with residual
        output = self.out_proj(self.dropout(fused))
        output = self.layer_norm(output + shared_output)  # Residual from shared

        return output


class GatedResidualFusion(nn.Module):
    """Gate-based fusion: gate * shared + (1-gate) * private.

    Simpler alternative to attention fusion. The gate is computed as:
        gate = sigmoid(W_g @ [h_shared; h_private])
        h_fused = gate * h_shared + (1 - gate) * h_private
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """Initialize gated fusion module.

        Args:
            hidden_size: Hidden dimension size
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Gate computation
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
            nn.Sigmoid(),
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        shared_output: torch.Tensor,
        private_output: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse shared and private outputs via gating.

        Args:
            shared_output: [batch_size, seq_len, hidden_size]
            private_output: [batch_size, seq_len, hidden_size]

        Returns:
            Fused output: [batch_size, seq_len, hidden_size]
        """
        # Concatenate shared and private
        concat = torch.cat([shared_output, private_output], dim=-1)  # [B, L, 2H]

        # Compute gate
        gate = self.gate_proj(concat)  # [B, L, H]

        # Weighted combination
        fused = gate * shared_output + (1 - gate) * private_output

        # Layer norm
        output = self.layer_norm(fused)

        return output


class SharedPrivateLoRA(nn.Module):
    """Flat Shared-Private adapter architecture (S3a).

    Architecture:
        - shared_adapter: LoRA(r=16) applied to all attention layers
        - private_adapters: Dict[task_name, LoRA(r=8)] per task
        - fusion: AttentionFusion or GatedResidualFusion

    This is the main architectural component being tested against ablations.
    """

    def __init__(
        self,
        base_model: nn.Module,
        task_names: List[str],
        shared_rank: int = 16,
        private_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        fusion_type: Literal["attention", "gated", "none"] = "attention",
    ):
        """Initialize shared-private LoRA architecture.

        Args:
            base_model: Base pretrained model
            task_names: List of task names
            shared_rank: Rank for shared LoRA (default: 16)
            private_rank: Rank for private LoRA per task (default: 8)
            lora_alpha: LoRA alpha scaling parameter
            lora_dropout: Dropout for LoRA layers
            target_modules: Which modules to apply LoRA to
            fusion_type: Type of fusion ("attention", "gated", or "none")
        """
        super().__init__()
        self.task_names = task_names
        self.fusion_type = fusion_type

        # Default target modules for common architectures
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

        # Store base model
        self.base_model = base_model

        # Create shared LoRA adapter
        shared_config = LoraConfig(
            r=shared_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.shared_adapter = get_peft_model(base_model, shared_config)

        # Create private adapters per task
        # Note: In practice, we'd use PEFT's adapter switching mechanism
        # For now, we'll create separate adapter configs
        self.private_adapters = nn.ModuleDict()
        for task_name in task_names:
            private_config = LoraConfig(
                r=private_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            # Store config for later instantiation
            self.private_adapters[task_name] = nn.Identity()  # Placeholder
            # TODO: Implement proper adapter switching with PEFT

        # Create fusion module
        if fusion_type == "attention":
            hidden_size = base_model.config.hidden_size
            self.fusion = AttentionFusion(hidden_size)
        elif fusion_type == "gated":
            hidden_size = base_model.config.hidden_size
            self.fusion = GatedResidualFusion(hidden_size)
        elif fusion_type == "none":
            self.fusion = None
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with shared-private adapter fusion.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            task: Task name for selecting private adapter
            **kwargs: Additional arguments for base model

        Returns:
            Hidden states [batch_size, seq_len, hidden_size]
        """
        # Get shared adapter output
        shared_output = self.shared_adapter(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        shared_hidden = shared_output.hidden_states[-1]

        # Get private adapter output for the specific task
        # TODO: Implement proper adapter switching
        # For now, we'll use the same output as private
        # In practice, this would switch to the task-specific adapter
        private_hidden = shared_hidden  # Placeholder

        # Fuse outputs
        if self.fusion is not None:
            fused_hidden = self.fusion(shared_hidden, private_hidden)
        else:
            # No fusion: simple concatenation or addition
            fused_hidden = shared_hidden + private_hidden

        return fused_hidden

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return count_trainable_params(self)


def count_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_parameter_parity(
    models: Dict[str, nn.Module],
    tolerance: float = 0.05,
) -> bool:
    """Verify that all models have similar trainable parameter counts.

    CRITICAL: This enforces the parameter parity requirement for ablations.
    All variants (A1-A4) must have comparable trainable parameters.

    Args:
        models: Dictionary mapping model names to models
        tolerance: Maximum allowed deviation (default: 5%)

    Returns:
        True if all models are within tolerance

    Raises:
        AssertionError if parameter counts differ by more than tolerance
    """
    param_counts = {
        name: count_trainable_params(model)
        for name, model in models.items()
    }

    # Compute mean and deviations
    mean_params = sum(param_counts.values()) / len(param_counts)
    deviations = {
        name: abs(count - mean_params) / mean_params
        for name, count in param_counts.items()
    }

    # Check tolerance
    max_deviation = max(deviations.values())
    max_deviation_model = max(deviations, key=deviations.get)

    print(f"\n{'='*60}")
    print("PARAMETER PARITY CHECK")
    print(f"{'='*60}")
    for name, count in param_counts.items():
        deviation = deviations[name] * 100
        status = "✓" if deviation <= tolerance * 100 else "✗"
        print(f"{status} {name:10s}: {count:>12,} params ({deviation:>5.2f}% from mean)")
    print(f"  Mean: {mean_params:>12,.0f} params")
    print(f"  Max deviation: {max_deviation*100:.2f}% ({max_deviation_model})")
    print(f"  Tolerance: {tolerance*100:.0f}%")

    if max_deviation > tolerance:
        raise AssertionError(
            f"Parameter parity violated! {max_deviation_model} deviates by "
            f"{max_deviation*100:.2f}% (tolerance: {tolerance*100:.0f}%)"
        )

    print(f"\n✓ Parameter parity check PASSED")
    print(f"{'='*60}\n")

    return True


def create_shared_lora_config(
    rank: int,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """Create LoRA config for shared adapter.

    Args:
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        target_modules: Target modules (default: attention + MLP)

    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )


def create_private_lora_config(
    rank: int,
    alpha: int = 32,
    dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    """Create LoRA config for private (task-specific) adapter.

    Args:
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        target_modules: Target modules (default: attention + MLP)

    Returns:
        LoraConfig object
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )


def estimate_lora_params(
    base_model_params: int,
    rank: int,
    num_target_modules: int = 7,  # q, k, v, o, gate, up, down
    alpha: int = 32,
) -> int:
    """Estimate number of trainable parameters for LoRA.

    Rough approximation:
        params_per_layer = 2 * hidden_size * rank * num_target_modules
        total_params = params_per_layer * num_layers

    Args:
        base_model_params: Number of parameters in base model
        rank: LoRA rank
        num_target_modules: Number of modules with LoRA applied
        alpha: LoRA alpha (doesn't affect param count)

    Returns:
        Estimated trainable parameters
    """
    # Rough heuristic based on typical transformer architectures
    # For a 3B model with 32 layers and hidden_size ~3072:
    # Each LoRA layer adds 2 * 3072 * r per target module
    # With 7 target modules per layer: 7 * 2 * 3072 * r * 32 layers

    # Estimate hidden size from param count
    # Very rough: hidden_size ≈ sqrt(params / 100)
    estimated_hidden = int((base_model_params / 100) ** 0.5)
    estimated_layers = 32 if base_model_params > 2e9 else 24

    params_per_layer = 2 * estimated_hidden * rank * num_target_modules
    total_params = params_per_layer * estimated_layers

    return total_params
