# src/CLAUDE.md — Source Code Instructions

## Architecture Overview

```
src/
├── data/          # Data loading and preprocessing
├── models/        # Model architectures and adapters
├── training/      # Training loops, optimizers, callbacks
├── evaluation/    # Metrics, analysis, contamination checks
├── quantization/  # QLoRA, GPTQ, AWQ pipelines
└── utils/         # Auto-batch, VRAM monitor, checkpointing
```

## Critical Implementation Rules

### 1. Token Counting Is Non-Negotiable
Every training run MUST log cumulative tokens seen per task. This is required for the token-controlled baseline (RQ5), which is the most important experimental control.

```python
# In every training loop:
self.tokens_seen = defaultdict(int)  # task_name -> total tokens

def on_step(self, batch, task_name):
    n_tokens = (batch['attention_mask'] != 0).sum().item()
    self.tokens_seen[task_name] += n_tokens
```

### 2. Parameter Parity for Ablations
Architecture ablations A1–A4 MUST have comparable trainable parameter counts (within 5%). Assert this before training:

```python
def assert_parameter_parity(models: dict, tolerance=0.05):
    counts = {name: sum(p.numel() for p in m.parameters() if p.requires_grad) 
              for name, m in models.items()}
    max_count = max(counts.values())
    for name, count in counts.items():
        ratio = count / max_count
        assert ratio > (1 - tolerance), f"{name} has {count} params ({ratio:.2%} of max)"
```

### 3. Checkpoint Every 200 Steps
Kaggle sessions die without warning. Save LoRA adapters (not full model) every 200 steps.

### 4. Smoke Test Before Every Run
Run 50 steps on 100 samples. If loss doesn't decrease by 50%, kill and debug.

### 5. Auto-Batch Before Every New Model
Use `utils/auto_batch.py` to find max batch size at 85% VRAM utilization.

### 6. SpanClassificationHead for Task 2015-T14
Standard BIO tagging cannot handle discontiguous entities. Use span-based classification.

## Coding Standards

- PyTorch 2.2+, Python 3.10+
- Type hints on all function signatures
- Docstrings on all public methods
- Use `@dataclass` for configs and data containers
- Log everything to wandb
- Save results as JSON after every experiment
- Use pathlib, not os.path
