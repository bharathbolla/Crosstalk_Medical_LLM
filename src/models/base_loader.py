"""Model loading with automatic quantization and optimization.

Supports:
- 6 base models (Phi-3, Gemma-2, Llama-3.2, Qwen2.5)
- Auto-quantization (QLoRA 4-bit for >4B params, FP16 for <=4B)
- Gradient checkpointing for memory efficiency
- Flash Attention 2 with graceful fallback
- PEFT/LoRA adapter setup
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get HuggingFace token
HF_TOKEN = os.getenv("HF_TOKEN", None)


# Model registry with HuggingFace IDs
MODEL_REGISTRY = {
    "phi3_mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi3_small": "microsoft/Phi-3-small-8k-instruct",
    "gemma2_2b": "google/gemma-2-2b",
    "gemma2_9b": "google/gemma-2-9b",
    "llama32_3b": "meta-llama/Llama-3.2-3B",
    "qwen25_7b": "Qwen/Qwen2.5-7B",
}

# Model sizes (billions of parameters)
MODEL_SIZES = {
    "phi3_mini": 3.8,
    "phi3_small": 7.0,
    "gemma2_2b": 2.0,
    "gemma2_9b": 9.0,
    "llama32_3b": 3.0,
    "qwen25_7b": 7.0,
}


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name: str
    model_id: str
    model_size_b: float
    use_quantization: bool
    quantization_bits: int
    use_flash_attn: bool
    use_gradient_checkpointing: bool
    device_map: str
    torch_dtype: torch.dtype


def get_model_config(
    model_name: str,
    force_quantization: Optional[bool] = None,
    quantization_bits: int = 4,
    use_flash_attn: bool = True,
    use_gradient_checkpointing: bool = True,
) -> ModelConfig:
    """Get model configuration with auto-quantization decision.

    Args:
        model_name: Model name from MODEL_REGISTRY
        force_quantization: Force quantization on/off (None = auto-decide)
        quantization_bits: Quantization precision (4 or 8)
        use_flash_attn: Whether to try Flash Attention 2
        use_gradient_checkpointing: Whether to enable gradient checkpointing

    Returns:
        ModelConfig object
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_id = MODEL_REGISTRY[model_name]
    model_size = MODEL_SIZES[model_name]

    # Auto-decide quantization: use QLoRA for models >4B params
    if force_quantization is None:
        use_quantization = model_size > 4.0
    else:
        use_quantization = force_quantization

    # Auto-decide gradient checkpointing: use for models >6B
    if use_gradient_checkpointing is None:
        use_gradient_checkpointing = model_size > 6.0

    # Determine torch dtype
    if use_quantization:
        torch_dtype = torch.float16  # BitsAndBytes requires FP16
    else:
        torch_dtype = torch.float16  # Use FP16 for efficiency

    return ModelConfig(
        model_name=model_name,
        model_id=model_id,
        model_size_b=model_size,
        use_quantization=use_quantization,
        quantization_bits=quantization_bits,
        use_flash_attn=use_flash_attn,
        use_gradient_checkpointing=use_gradient_checkpointing,
        device_map="auto",
        torch_dtype=torch_dtype,
    )


def load_model(
    model_name: str,
    task_type: Literal["causal_lm", "sequence_classification", "token_classification"] = "causal_lm",
    num_labels: Optional[int] = None,
    lora_config: Optional[LoraConfig] = None,
    force_quantization: Optional[bool] = None,
    quantization_bits: int = 4,
    use_flash_attn: bool = True,
    use_gradient_checkpointing: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, AutoTokenizer, ModelConfig]:
    """Load a model with automatic optimization.

    Args:
        model_name: Model name from MODEL_REGISTRY
        task_type: Type of task (causal_lm, sequence_classification, token_classification)
        num_labels: Number of labels (required for classification tasks)
        lora_config: LoRA configuration (if None, no adapter applied)
        force_quantization: Force quantization on/off (None = auto-decide)
        quantization_bits: Quantization precision (4 or 8)
        use_flash_attn: Whether to try Flash Attention 2
        use_gradient_checkpointing: Whether to enable gradient checkpointing
        device: Device to load model on (None = auto)

    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Get model configuration
    config = get_model_config(
        model_name=model_name,
        force_quantization=force_quantization,
        quantization_bits=quantization_bits,
        use_flash_attn=use_flash_attn,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )

    # Set environment variables for optimal memory allocation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Prepare quantization config
    quantization_config = None
    if config.use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(quantization_bits == 4),
            load_in_8bit=(quantization_bits == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Prepare model loading kwargs
    model_kwargs = {
        "pretrained_model_name_or_path": config.model_id,
        "torch_dtype": config.torch_dtype,
        "device_map": config.device_map,
        "trust_remote_code": True,
        "quantization_config": quantization_config,
        "token": HF_TOKEN,  # HuggingFace token for gated models
    }

    # Try Flash Attention 2
    if config.use_flash_attn:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            print("⚠ Flash Attention 2 not available, using default attention")
            pass

    # Load model based on task type
    if task_type == "causal_lm":
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    elif task_type == "sequence_classification":
        if num_labels is None:
            raise ValueError("num_labels required for sequence_classification")
        model_kwargs["num_labels"] = num_labels
        model = AutoModelForSequenceClassification.from_pretrained(**model_kwargs)
    elif task_type == "token_classification":
        if num_labels is None:
            raise ValueError("num_labels required for token_classification")
        model_kwargs["num_labels"] = num_labels
        model = AutoModelForTokenClassification.from_pretrained(**model_kwargs)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # Enable gradient checkpointing
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Apply LoRA if config provided
    if lora_config is not None:
        model = get_peft_model(model, lora_config)
        print(f"✓ Applied LoRA with config: {lora_config}")
        print(f"  Trainable params: {model.print_trainable_parameters()}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        padding_side="left",  # For causal LM
        token=HF_TOKEN,  # HuggingFace token for gated models
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, "config"):
            model.config.pad_token_id = tokenizer.eos_token_id

    # Move to device if specified and not using device_map="auto"
    if device is not None and config.device_map != "auto":
        model = model.to(device)

    # Print summary
    print(f"\n{'='*60}")
    print(f"MODEL LOADED: {config.model_name}")
    print(f"{'='*60}")
    print(f"Model ID:      {config.model_id}")
    print(f"Size:          {config.model_size_b:.1f}B parameters")
    print(f"Quantization:  {quantization_bits}-bit" if config.use_quantization else "FP16 (no quantization)")
    print(f"Flash Attn:    {'Enabled' if config.use_flash_attn else 'Disabled'}")
    print(f"Grad Ckpt:     {'Enabled' if config.use_gradient_checkpointing else 'Disabled'}")
    print(f"Device:        {next(model.parameters()).device}")
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"VRAM:          {mem_allocated:.2f} GB")
    print(f"{'='*60}\n")

    return model, tokenizer, config


def create_lora_config(
    task_type: TaskType = TaskType.CAUSAL_LM,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[list] = None,
    bias: str = "none",
) -> LoraConfig:
    """Create a LoRA configuration.

    Args:
        task_type: PEFT task type
        r: LoRA rank (lower = fewer params, less capacity)
        lora_alpha: LoRA alpha (scaling factor)
        lora_dropout: Dropout probability for LoRA layers
        target_modules: Which modules to apply LoRA to (None = auto-detect)
        bias: Bias training strategy ("none", "all", "lora_only")

    Returns:
        LoraConfig object
    """
    # Auto-detect target modules for common architectures
    if target_modules is None:
        # Common attention projection layers across architectures
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    return LoraConfig(
        task_type=task_type,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        inference_mode=False,
    )


def estimate_model_memory(model_size_b: float, quantization_bits: Optional[int] = None) -> float:
    """Estimate model memory usage in GB.

    Args:
        model_size_b: Model size in billions of parameters
        quantization_bits: Quantization precision (None = FP16)

    Returns:
        Estimated memory in GB
    """
    params = model_size_b * 1e9

    if quantization_bits == 4:
        bytes_per_param = 0.5  # 4-bit quantization
    elif quantization_bits == 8:
        bytes_per_param = 1.0  # 8-bit quantization
    else:
        bytes_per_param = 2.0  # FP16

    # Add overhead for activations, optimizer states, etc. (rough estimate)
    overhead_multiplier = 1.5

    memory_gb = (params * bytes_per_param * overhead_multiplier) / (1024 ** 3)

    return memory_gb
