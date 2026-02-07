"""Model loading and architecture components."""

from .base_loader import load_model, get_model_config, create_lora_config
from .adapters import (
    SharedPrivateLoRA,
    AttentionFusion,
    GatedResidualFusion,
    count_trainable_params,
    verify_parameter_parity,
    create_shared_lora_config,
    create_private_lora_config,
)
from .hierarchical import (
    HierarchicalMTLModel,
    CrossLevelAttention,
    HierarchicalAdapter,
)
from .heads import (
    TokenClassificationHead,
    SpanClassificationHead,
    RelationExtractionHead,
    SequenceRankingHead,
    create_task_head,
)
from .multitask_model import MultiTaskModel
from .ablations import (
    create_ablation_variant,
    create_all_ablations,
    calculate_ablation_ranks,
    print_ablation_summary,
)

__all__ = [
    # Base loader
    "load_model",
    "get_model_config",
    "create_lora_config",
    # Adapters
    "SharedPrivateLoRA",
    "AttentionFusion",
    "GatedResidualFusion",
    "count_trainable_params",
    "verify_parameter_parity",
    "create_shared_lora_config",
    "create_private_lora_config",
    # Hierarchical
    "HierarchicalMTLModel",
    "CrossLevelAttention",
    "HierarchicalAdapter",
    # Heads
    "TokenClassificationHead",
    "SpanClassificationHead",
    "RelationExtractionHead",
    "SequenceRankingHead",
    "create_task_head",
    # Multi-task model
    "MultiTaskModel",
    # Ablations
    "create_ablation_variant",
    "create_all_ablations",
    "calculate_ablation_ranks",
    "print_ablation_summary",
]
