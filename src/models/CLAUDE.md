# src/models/CLAUDE.md — Model Architecture Instructions

## Architecture Variants

This project tests 6 training strategies and 4 architectural ablations. Every architecture choice must be empirically justified — no hand-waving.

## Strategy → Architecture Mapping

| Strategy | Architecture | Gradient Mgmt | Key File |
|---|---|---|---|
| S1 | Single LoRA per task | Standard | adapters.py |
| S2 | Shared LoRA across tasks | Standard | adapters.py |
| S3a | Flat Shared (r=16) + Private (r=8) + Fusion | PCGrad | adapters.py |
| S3b | Hierarchical: Level1 → Level2 cascade | PCGrad | hierarchical.py |
| S4 | Sequential: pretrain shared → adapt per-task | Standard | adapters.py |
| S5 | Best of S3a/S3b with QLoRA base | As above | qlora.py + above |

## Architectural Ablation Matrix (A1–A4)

**PARAMETER PARITY IS MANDATORY.** All variants must have comparable trainable params (within 5%).

| Variant | Shared | Private | Fusion | Total ~Params | Purpose |
|---|---|---|---|---|---|
| A1 | r=24, all tasks | None | None | ~26M | Lower bound |
| A2 | None | r=24, per task | None | ~26M | Upper bound (S1 equivalent) |
| A3 | r=16 | r=8 | None (concat) | ~26M | Tests composition without fusion |
| A4 | r=16 | r=8 | Attention | ~26M | Full model (our proposal) |

Adjust ranks so trainable params match. Assert before training:

```python
# In run_experiment.py, before training starts:
assert_parameter_parity({"A1": model_a1, "A2": model_a2, "A3": model_a3, "A4": model_a4})
```

## base_loader.py

```python
def load_model(
    model_name: str,
    task_type: str = "ner",
    quantization: Optional[dict] = None,
    gradient_checkpointing: bool = True,
) -> tuple[PreTrainedModel, bool]:
    """
    Load model with auto-quantization based on T4 VRAM constraints.
    
    Decision logic:
    - params > 4B → QLoRA NF4
    - params ≤ 4B → FP16 LoRA
    - Always enable gradient_checkpointing for 7B+
    - Always try flash_attention_2
    
    Returns: (model, is_quantized)
    """
```

## adapters.py — Shared-Private LoRA

```python
class SharedPrivateLoRA(nn.Module):
    """
    Flat Shared-Private adapter for S3a.
    
    - shared_adapter: LoRA(r=16) applied to all attention layers
    - private_adapters: dict of LoRA(r=8) per task
    - fusion: AttentionFusion or GatedResidual (ablate both)
    
    CRITICAL: Track trainable params and assert parity with A1-A4.
    """
    
class AttentionFusion(nn.Module):
    """
    Learns to weight shared vs private adapter outputs.
    h_fused = softmax(W_q @ h_shared, W_k @ h_private) @ [h_shared; h_private]
    
    Alternative: GatedResidual
    h_fused = gate * h_shared + (1 - gate) * h_private
    where gate = sigmoid(W_g @ [h_shared; h_private])
    """
```

## hierarchical.py — Hierarchical MTL (S3b)

```python
class HierarchicalMTLModel(nn.Module):
    """
    PRIMARY NOVELTY — Hierarchical Multi-Task Learning.
    
    Level 1 (Low-level): NER tasks
      - shared_adapter → level1_adapter → NER head
      
    Level 2 (High-level): RE, QA, Temporal tasks
      - shared_adapter → level1_adapter (entity features)
      - cross_level_attention(shared_features, level1_features)
      - level2_adapter → task-specific head
    
    Key design choices to ablate:
    1. detach_l1_for_l2: Should L2 gradients flow back through L1?
       - True: L1 is treated as frozen feature extractor for L2
       - False: L2 training also improves L1 (but risk of interference)
    2. cross_level_attention vs simple concatenation
    3. With vs without PCGrad
    """
```

## heads.py — Task-Specific Heads

### SpanClassificationHead (for discontiguous entities)
```python
class SpanClassificationHead(nn.Module):
    """
    For Task 2015-T14. Enumerates candidate spans, classifies each,
    then links discontiguous spans of the same entity.
    
    Steps:
    1. Enumerate all spans up to max_span_len
    2. Create span representations: [start_rep; end_rep; width_embed]
    3. Classify each span (entity type or null)
    4. For non-null spans, compute pairwise linking scores
    5. Group linked spans into entities
    """
```

### TokenClassificationHead (standard NER)
```python
class TokenClassificationHead(nn.Module):
    """Standard BIO/BIOES tagging for Tasks 2014-T7, 2021-T6 NER."""
```

### RelationExtractionHead
```python
class RelationExtractionHead(nn.Module):
    """
    For Tasks 2016-T12, 2021-T6 RE.
    Input: hidden states + entity span markers
    Output: relation type logits for each entity pair
    """
```

### SequenceRankingHead
```python
class SequenceRankingHead(nn.Module):
    """
    For Task 2017-T3 QA ranking.
    Input: [CLS] representation of question-answer pair
    Output: relevance score
    Trained with pairwise ranking loss (margin-based)
    """
```

## Model Loading Checklist

Before loading ANY model, run:
1. `auto_batch.py` to find max batch size
2. Verify VRAM fits within T4 16GB
3. Enable gradient checkpointing for 7B+
4. Enable flash attention 2 if model supports it
5. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
