# configs/CLAUDE.md — Configuration System

## Overview

All experiments use Hydra for config management. Configs compose: model + task + strategy = experiment.

## Config Hierarchy

```
configs/
├── model/          # One YAML per model
├── task/           # One YAML per SemEval task
├── strategy/       # One YAML per training strategy (S1–S5, S3b)
└── experiment/     # Combined configs for reproducible runs
```

## Model Configs

Each model config specifies: HuggingFace ID, quantization settings, LoRA targets, and T4 compatibility.

### Key Decision: Quantization

```
if model_params > 4B:
    use QLoRA 4-bit on T4 (16 GB)
else:
    use FP16 LoRA on T4
```

### Model Registry

| Config File | Model | Params | T4 Method | LoRA rank | Target Modules |
|---|---|---|---|---|---|
| gemma2b.yaml | google/gemma-2-2b | 2.6B | FP16 LoRA | 16 | q,k,v,o |
| phi3mini.yaml | microsoft/Phi-3-mini-4k-instruct | 3.8B | FP16 LoRA | 16 | q,k,v,o |
| llama3b.yaml | meta-llama/Llama-3.2-3B | 3.2B | FP16 LoRA | 16 | q,k,v,o,gate,up,down |
| mistral7b.yaml | mistralai/Mistral-7B-v0.3 | 7.2B | QLoRA NF4 | 32 | q,k,v,o |
| llama8b.yaml | meta-llama/Llama-3.1-8B | 8.0B | QLoRA NF4 | 32 | q,k,v,o,gate,up,down |
| biomistral.yaml | BioMistral/BioMistral-7B | 7.2B | QLoRA NF4 | 32 | q,k,v,o |

### Example: configs/model/llama8b.yaml

```yaml
model:
  name: "meta-llama/Llama-3.1-8B"
  short_name: "llama8b"
  params: 8.0e9
  
  quantization:
    enabled: true
    bits: 4
    quant_type: "nf4"
    compute_dtype: "float16"
    double_quant: true
  
  lora:
    rank: 32
    alpha: 64
    dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  
  training:
    max_batch_size_t4: 4          # Known safe batch for T4 16GB
    gradient_checkpointing: true   # Always for 7B+
    flash_attention: true
    
  unsloth:
    compatible: true
    use_for: ["S1", "S2"]         # Only single/shared LoRA strategies
```

## Task Configs

### Example: configs/task/semeval2014t7.yaml

```yaml
task:
  name: "semeval2014t7"
  full_name: "SemEval 2014 Task 7 - Analysis of Clinical Text"
  type: "ner"
  task_level: 1                    # Level 1 = low-level (for hierarchical MTL)
  
  data:
    source: "physionet"            # requires credentialed access
    train_size: 9000
    dev_size: 1000
    test_size: 7000
    
  evaluation:
    primary_metric: "strict_f1"
    secondary_metrics: ["relaxed_f1", "accuracy"]
    eval_script: "official"        # use task's official eval script
    published_sota: 0.813
    sota_system: "UTH-CCB"
    
  preprocessing:
    max_seq_len: null              # Set by analyze_seq_lengths.py
    entity_handling: "bio"         # standard BIO tagging
    
  contamination:
    risk: "high"                   # public since 2014
    zero_shot_threshold: 0.57      # 70% of SOTA
```

### Task Level Assignment (for Hierarchical MTL S3b)

```
Level 1 (Low-level — entity identification):
  - semeval2014t7  (NER + Normalization)
  - semeval2015t14 (Disorder NER — discontiguous)
  - semeval2021t6  (NER component)

Level 2 (High-level — relation/reasoning):
  - semeval2016t12 (Temporal Reasoning)
  - semeval2017t3  (QA Ranking)
  - semeval2021t6  (RE component)
```

## Strategy Configs

### S1: Single-Task LoRA
```yaml
strategy:
  name: "S1_single_task"
  type: "single_task"
  description: "One LoRA adapter per task. Baseline."
  
  adapter:
    shared: false
    private: true
    
  training:
    epochs: 5
    early_stopping_patience: 3
    eval_every_n_steps: 200
    
  token_logging: true              # CRITICAL: log tokens for RQ5 control
```

### S2: Shared LoRA MTL
```yaml
strategy:
  name: "S2_shared_mtl"
  type: "multi_task"
  description: "Single shared LoRA adapter trained on all tasks."
  
  adapter:
    shared: true
    private: false
    
  multitask:
    sampling: "temperature"
    temperature: 2.0
    loss_weighting: "uncertainty"
    
  token_logging: true
```

### S3a: Flat Shared-Private + PCGrad
```yaml
strategy:
  name: "S3a_shared_private_flat"
  type: "multi_task"
  description: "Shared (r=16) + Private (r=8) adapters, flat combination."
  
  adapter:
    shared: true
    shared_rank: 16
    private: true
    private_rank: 8
    fusion: "attention"            # attention-weighted combination
    
  gradient:
    conflict_resolution: "pcgrad"  # Projected Conflicting Gradients
    
  multitask:
    sampling: "temperature"
    temperature: 2.0
    loss_weighting: "uncertainty"
```

### S3b: Hierarchical MTL + PCGrad (PRIMARY NOVELTY)
```yaml
strategy:
  name: "S3b_hierarchical_mtl"
  type: "multi_task"
  description: "Hierarchical: Level 1 (NER) feeds Level 2 (RE/QA). Primary novelty."
  
  architecture:
    type: "hierarchical"
    shared_rank: 16
    level1_rank: 8                 # entity-level features
    level2_rank: 8                 # relation-level features
    cross_level_attention: true
    detach_l1_for_l2: false        # allow gradients to flow; ablate this
    
  gradient:
    conflict_resolution: "pcgrad"
```

### S5: QLoRA 4-bit MTL
```yaml
strategy:
  name: "S5_qlora_mtl"
  type: "multi_task"
  description: "Best MTL architecture (S3a or S3b) with QLoRA 4-bit base."
  
  quantization:
    enabled: true
    bits: 4
    quant_type: "nf4"
    
  # Inherits adapter config from best of S3a/S3b
  architecture:
    inherit_from: "best_s3"        # decided after Phase 3
```

## Architecture Ablation Configs (A1–A4)

**CRITICAL**: All variants MUST have comparable trainable parameter counts.

```yaml
# configs/experiment/ablation_architecture.yaml
ablations:
  A1_shared_only:
    shared_rank: 24                # increased rank to match param count
    private: false
    fusion: null
    trainable_params: "~26M"       # target: match A4
    
  A2_private_only:
    shared: false
    private_rank: 24               # increased rank to match param count  
    fusion: null
    trainable_params: "~26M"
    
  A3_shared_private_no_fusion:
    shared_rank: 16
    private_rank: 8
    fusion: null                   # simple concatenation instead
    trainable_params: "~26M"
    
  A4_shared_private_fusion:
    shared_rank: 16
    private_rank: 8
    fusion: "attention"
    trainable_params: "~26M"       # includes fusion layer params
    
  parameter_parity_check: true     # ASSERT all variants within 5% of target
```

## Token-Controlled Baseline Config (RQ5 — CRITICAL)

```yaml
# configs/experiment/token_controlled.yaml
token_control:
  description: >
    Single-task models trained with the SAME total number of tokens
    as multi-task models. Multi-task sees tokens from all 5 tasks;
    token-controlled single-task sees the same count by oversampling
    its own task data.
    
  method: "oversample_single_task"
  
  # After multi-task training, record total_tokens_seen
  # Then train single-task with that exact count
  match_to: "S3b_total_tokens"    # match to best multi-task variant
  
  logging:
    log_tokens_per_step: true
    log_tokens_per_task: true
    log_cumulative_tokens: true
```

## Hyperparameter Search Space

```yaml
hp_search:
  learning_rate: [1e-5, 2e-5, 5e-5, 1e-4]
  lora_rank: [8, 16, 32]          # reduced from [8,16,32,64] to save GPU
  batch_size: [16, 32]            # effective, via gradient accumulation
  epochs: [3, 5]                  # with early stopping
  warmup_ratio: [0.05, 0.1]
  
  # Multi-task specific
  sampling_strategy: ["proportional", "uniform", "temperature"]
  temperature: [1.0, 2.0, 5.0]
  loss_weighting: ["equal", "uncertainty"]
  
  # Only top 3 models get full search; others get default
  full_search_models: 3
  reduced_search_configs: 4        # for remaining models
```
