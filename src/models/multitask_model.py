"""Multi-task model wrapper that routes inputs to task-specific heads.

Supports all 6 training strategies (S1-S5, S3b) via configuration.
"""

from typing import Dict, Optional, Literal, Any

import torch
import torch.nn as nn

from .adapters import SharedPrivateLoRA, count_trainable_params
from .hierarchical import HierarchicalMTLModel
from .heads import (
    TokenClassificationHead,
    SpanClassificationHead,
    RelationExtractionHead,
    SequenceRankingHead,
    create_task_head,
)


class MultiTaskModel(nn.Module):
    """Multi-task model that wraps base model + adapters + task heads.

    Supports multiple strategies:
        - S1: Single-task LoRA (one model per task)
        - S2: Shared LoRA (one adapter for all tasks)
        - S3a: Flat Shared-Private LoRA with fusion
        - S3b: Hierarchical MTL with cross-level attention
        - S4: Sequential transfer (pretrain shared, then adapt per-task)
        - S5: QLoRA 4-bit quantization with best architecture

    Routes inputs to appropriate task heads based on task_type.
    """

    def __init__(
        self,
        base_model: nn.Module,
        strategy: Literal["S1", "S2", "S3a", "S3b", "S4", "S5"],
        task_configs: Dict[str, Dict[str, Any]],
        adapter_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize multi-task model.

        Args:
            base_model: Base pretrained model
            strategy: Training strategy (S1-S5)
            task_configs: Dictionary mapping task names to configs
                         Each config should have:
                         - task_type: "ner", "span", "re", or "qa"
                         - num_labels: Number of labels/classes
                         - Additional head-specific params
            adapter_config: Optional adapter configuration
        """
        super().__init__()
        self.strategy = strategy
        self.task_configs = task_configs
        self.hidden_size = base_model.config.hidden_size

        # Initialize adapter based on strategy
        if strategy == "S1":
            # Single-task: just use base model with LoRA per task
            self.adapter = None
            self.base_model = base_model

        elif strategy == "S2":
            # Shared LoRA: one adapter for all tasks
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=adapter_config.get("rank", 16) if adapter_config else 16,
                lora_alpha=adapter_config.get("alpha", 32) if adapter_config else 32,
                lora_dropout=adapter_config.get("dropout", 0.1) if adapter_config else 0.1,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.adapter = get_peft_model(base_model, lora_config)
            self.base_model = self.adapter

        elif strategy == "S3a":
            # Flat Shared-Private with fusion
            task_names = list(task_configs.keys())
            self.adapter = SharedPrivateLoRA(
                base_model=base_model,
                task_names=task_names,
                shared_rank=adapter_config.get("shared_rank", 16) if adapter_config else 16,
                private_rank=adapter_config.get("private_rank", 8) if adapter_config else 8,
                lora_alpha=adapter_config.get("alpha", 32) if adapter_config else 32,
                lora_dropout=adapter_config.get("dropout", 0.1) if adapter_config else 0.1,
                fusion_type=adapter_config.get("fusion_type", "attention") if adapter_config else "attention",
            )
            self.base_model = self.adapter

        elif strategy == "S3b":
            # Hierarchical MTL
            self.adapter = HierarchicalMTLModel(
                base_model=base_model,
                shared_rank=adapter_config.get("shared_rank", 16) if adapter_config else 16,
                level1_rank=adapter_config.get("level1_rank", 8) if adapter_config else 8,
                level2_rank=adapter_config.get("level2_rank", 8) if adapter_config else 8,
                lora_alpha=adapter_config.get("alpha", 32) if adapter_config else 32,
                lora_dropout=adapter_config.get("dropout", 0.1) if adapter_config else 0.1,
                use_cross_attention=adapter_config.get("use_cross_attention", True) if adapter_config else True,
                detach_l1_for_l2=adapter_config.get("detach_l1_for_l2", False) if adapter_config else False,
            )
            self.base_model = self.adapter

        elif strategy == "S4":
            # Sequential transfer: similar to S2 but trained in phases
            # Phase 1: Pretrain shared adapter
            # Phase 2: Add task-specific adapters
            # For now, use same architecture as S3a
            task_names = list(task_configs.keys())
            self.adapter = SharedPrivateLoRA(
                base_model=base_model,
                task_names=task_names,
                shared_rank=adapter_config.get("shared_rank", 16) if adapter_config else 16,
                private_rank=adapter_config.get("private_rank", 8) if adapter_config else 8,
                lora_alpha=adapter_config.get("alpha", 32) if adapter_config else 32,
                lora_dropout=adapter_config.get("dropout", 0.1) if adapter_config else 0.1,
                fusion_type="none",  # No fusion for sequential
            )
            self.base_model = self.adapter

        elif strategy == "S5":
            # QLoRA 4-bit: use best architecture (S3a or S3b) with quantization
            # Quantization is handled in base_loader.py
            # Architecture defaults to S3b
            self.adapter = HierarchicalMTLModel(
                base_model=base_model,
                shared_rank=adapter_config.get("shared_rank", 16) if adapter_config else 16,
                level1_rank=adapter_config.get("level1_rank", 8) if adapter_config else 8,
                level2_rank=adapter_config.get("level2_rank", 8) if adapter_config else 8,
                lora_alpha=adapter_config.get("alpha", 32) if adapter_config else 32,
                lora_dropout=adapter_config.get("dropout", 0.1) if adapter_config else 0.1,
                use_cross_attention=adapter_config.get("use_cross_attention", True) if adapter_config else True,
                detach_l1_for_l2=adapter_config.get("detach_l1_for_l2", False) if adapter_config else False,
            )
            self.base_model = self.adapter

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Initialize task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, task_config in task_configs.items():
            head = create_task_head(
                task_type=task_config["task_type"],
                hidden_size=self.hidden_size,
                num_labels=task_config["num_labels"],
                **task_config.get("head_params", {}),
            )
            self.task_heads[task_name] = head

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with task routing.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            task: Task name (determines head selection)
            labels: Task-specific labels (optional)
            **kwargs: Additional task-specific arguments

        Returns:
            Dictionary with:
                - logits: Task-specific predictions
                - loss: Task loss (if labels provided)
                - hidden_states: Encoder hidden states (optional)
        """
        if task not in self.task_heads:
            raise ValueError(f"Unknown task: {task}. Available: {list(self.task_heads.keys())}")

        # Get encoder hidden states
        if self.strategy == "S3b":
            # Hierarchical: route based on task level
            hidden_states = self.adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
                **kwargs
            )
        elif self.strategy == "S3a":
            # Shared-Private: pass task name for private adapter selection
            hidden_states = self.adapter(
                input_ids=input_ids,
                attention_mask=attention_mask,
                task=task,
                **kwargs
            )
        else:
            # S1, S2, S4: standard forward
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )
            hidden_states = outputs.hidden_states[-1]

        # Route to task-specific head
        task_head = self.task_heads[task]
        task_config = self.task_configs[task]
        task_type = task_config["task_type"]

        if task_type == "ner":
            logits, loss = task_head(hidden_states, labels)
            return {"logits": logits, "loss": loss}

        elif task_type == "span":
            # Span classification requires span labels and link labels
            span_labels = kwargs.get("span_labels")
            link_labels = kwargs.get("link_labels")
            span_logits, link_scores, loss = task_head(
                hidden_states, span_labels, link_labels
            )
            return {
                "span_logits": span_logits,
                "link_scores": link_scores,
                "loss": loss,
            }

        elif task_type == "re":
            # Relation extraction requires entity positions
            head_start = kwargs.get("head_start")
            head_end = kwargs.get("head_end")
            tail_start = kwargs.get("tail_start")
            tail_end = kwargs.get("tail_end")
            logits, loss = task_head(
                hidden_states, head_start, head_end, tail_start, tail_end, labels
            )
            return {"logits": logits, "loss": loss}

        elif task_type == "qa":
            logits, loss = task_head(hidden_states, labels)
            return {"logits": logits, "loss": loss}

        else:
            raise ValueError(f"Unknown task_type: {task_type}")

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return count_trainable_params(self)

    def get_params_by_component(self) -> Dict[str, int]:
        """Get parameter counts by component.

        Returns:
            Dictionary with keys: "adapter", "heads", "total"
        """
        adapter_params = 0
        if self.adapter is not None:
            adapter_params = count_trainable_params(self.adapter)

        head_params = {
            task_name: count_trainable_params(head)
            for task_name, head in self.task_heads.items()
        }

        total_params = self.get_trainable_params()

        return {
            "adapter": adapter_params,
            "heads": head_params,
            "total": total_params,
        }

    def freeze_adapter(self):
        """Freeze adapter parameters (for S4 sequential training)."""
        if self.adapter is not None:
            for param in self.adapter.parameters():
                param.requires_grad = False

    def unfreeze_adapter(self):
        """Unfreeze adapter parameters."""
        if self.adapter is not None:
            for param in self.adapter.parameters():
                param.requires_grad = True

    def freeze_task_heads(self, exclude_tasks: Optional[list] = None):
        """Freeze task head parameters.

        Args:
            exclude_tasks: Tasks to NOT freeze (default: freeze all)
        """
        for task_name, head in self.task_heads.items():
            if exclude_tasks and task_name in exclude_tasks:
                continue
            for param in head.parameters():
                param.requires_grad = False

    def unfreeze_task_heads(self):
        """Unfreeze all task head parameters."""
        for head in self.task_heads.values():
            for param in head.parameters():
                param.requires_grad = True
