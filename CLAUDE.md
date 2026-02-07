# CLAUDE.md — Master Orchestration File
# Medical Cross-Task Knowledge Transfer with Small Language Models
# Version 3.0 — Incorporates all critique rounds

## Project Overview

This project studies whether **cross-task knowledge transfer** (not just multi-task learning) can enable small language models (2B–8B) to close a substantial fraction (60–85%) of the performance gap to proprietary LLMs on medical NLP tasks, under realistic deployment constraints.

**Critical distinction**: We explicitly separate gains from (a) increased data exposure from (b) genuine cross-task inductive bias. This is enforced via a **token-controlled baseline** that is the single most important experimental control.

## Core Falsifiable Claim

> If multi-task models outperform single-task models **even when total training tokens are equalized**, this constitutes evidence of genuine cross-task knowledge transfer rather than mere data exposure.

## Research Questions (v3.0)

- **RQ1**: After controlling for data contamination, does multi-task fine-tuning improve per-task performance compared to single-task fine-tuning for models under 8B parameters?
- **RQ2**: Can shared adapters learn representations that improve *out-of-distribution task performance* relative to parameter-matched single-task baselines?
- **RQ3**: What is the Pareto-optimal tradeoff between model size, quantization level, and task performance under deployment-equivalent settings (including calibration via ECE)?
- **RQ4**: Under what conditions does *negative transfer* occur, and can it be predicted from task similarity metrics (label schema overlap, vocabulary overlap)?
- **RQ5 (NEW — Critical Control)**: Does controlling for total training tokens eliminate multi-task gains? If gains persist under token parity, this supports genuine cross-task transfer.

## Directory Structure

```
medical-cross-task-transfer/
├── CLAUDE.md                          # THIS FILE — master orchestration
├── configs/
│   ├── CLAUDE.md                      # Config system instructions
│   ├── model/                         # Per-model YAML
│   ├── task/                          # Per-task YAML
│   ├── strategy/                      # S1–S5, S3b strategy configs
│   └── experiment/                    # Combined experiment configs
├── src/
│   ├── CLAUDE.md                      # Source code instructions
│   ├── data/
│   │   ├── CLAUDE.md                  # Data pipeline instructions
│   │   ├── base.py                    # UnifiedSample dataclass
│   │   ├── semeval2014t7.py           # Clinical Text Analysis parser
│   │   ├── semeval2015t14.py          # Disorder ID (discontiguous NER)
│   │   ├── semeval2016t12.py          # Clinical TempEval parser
│   │   ├── semeval2017t3.py           # Medical QA parser
│   │   ├── semeval2021t6.py           # Clinical NER+RE parser
│   │   ├── multitask_loader.py        # MultiTaskBatchSampler
│   │   └── collators.py              # Task-specific collators
│   ├── models/
│   │   ├── CLAUDE.md                  # Model architecture instructions
│   │   ├── base_loader.py            # Model loading with auto-quantization
│   │   ├── adapters.py               # SharedPrivateLoRA, AdapterFusion
│   │   ├── hierarchical.py           # HierarchicalMTLModel (S3b)
│   │   ├── heads.py                  # SpanClassificationHead, REHead, QAHead
│   │   └── multitask_model.py        # Main multi-task model
│   ├── training/
│   │   ├── CLAUDE.md                  # Training instructions
│   │   ├── trainer.py                # MultiTaskTrainer
│   │   ├── pcgrad.py                 # PCGrad optimizer
│   │   ├── loss.py                   # Uncertainty-weighted multi-task loss
│   │   └── callbacks.py             # Monitoring callbacks
│   ├── evaluation/
│   │   ├── CLAUDE.md                  # Evaluation instructions
│   │   ├── metrics.py               # Task metrics, bootstrap CI
│   │   ├── contamination.py         # 3-layer contamination checker
│   │   ├── calibration.py           # ECE computation
│   │   ├── probing.py               # Linear probes for adapter analysis
│   │   ├── transfer_analysis.py     # Transfer matrix, similarity scores
│   │   └── error_analysis.py        # 6-category error taxonomy
│   ├── quantization/
│   │   ├── qlora.py                 # QLoRA training setup
│   │   ├── gptq_export.py           # Post-training GPTQ
│   │   └── awq_export.py            # Post-training AWQ
│   └── utils/
│       ├── auto_batch.py            # Auto batch size finder
│       ├── vram_monitor.py          # VRAM leak detection
│       ├── checkpoint.py            # Kaggle-safe checkpoint manager
│       └── smoke_test.py            # Quick validation before long runs
├── scripts/
│   ├── run_experiment.py            # Main entry point (Hydra)
│   ├── run_contamination_check.py   # Phase 0 contamination audit
│   ├── run_probing.py               # Adapter probing tasks
│   ├── run_baseline.py              # BERT/BioBERT baselines
│   ├── run_gpt4_baseline.py         # GPT-4o-mini API evaluation
│   ├── run_inference_bench.py       # Latency/throughput profiling
│   └── generate_paper_tables.py     # Auto-generate LaTeX
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_contamination_report.ipynb
│   ├── 03_transfer_heatmap.ipynb
│   ├── 04_pareto_frontier.ipynb
│   ├── 05_probing_results.ipynb
│   ├── 06_calibration_plots.ipynb
│   ├── 07_error_analysis.ipynb
│   └── 08_token_controlled_analysis.ipynb  # NEW: RQ5 analysis
├── requirements.txt
├── setup.py
└── README.md
```

## Execution Order

### Phase 0: Setup & Contamination (Weeks 1–3)
1. Apply for PhysioNet access (Day 1 — DO NOT DELAY)
2. Set up repo, install deps, configure wandb
3. Implement all 5 data parsers with unit tests
4. Run contamination check (all 6 models × 5 tasks)
5. Decision point: replace contaminated tasks if needed

### Phase 1: Baselines (Week 4)
1. BERT, BioBERT, ClinicalBERT on all 5 tasks (Kaggle free T4)
2. Reproduce published SOTA within 2 F1 points

### Phase 2: Single-Task (Weeks 5–7)
1. S1: LoRA fine-tune all 6 models × 5 tasks
2. HP search for top 3 models
3. **Token-count logging starts here** — log tokens/task for every run

### Phase 3: Multi-Task (Weeks 8–10)
1. S2: Shared LoRA multi-task
2. S3a: Flat Shared-Private + PCGrad
3. S3b: Hierarchical MTL + PCGrad
4. S4: Sequential transfer
5. **Token-controlled baseline**: S1 models trained with same total tokens as S2/S3
6. Architecture ablation (A1–A4)
7. MIMIC exclusion control
8. Transfer matrix + negative transfer analysis

### Phase 4: Quantization & Efficiency (Weeks 11–12)
1. S5: QLoRA 4-bit for best architecture
2. GPTQ/AWQ post-training quantization
3. GPT-4o-mini API baselines (framed as reference points)
4. Inference benchmarking
5. ECE calibration measurement

### Phase 5: Analysis & Writing (Weeks 13–16)
1. Probing tasks (4 probes on shared adapter)
2. Error analysis (6 categories)
3. Generate all figures and tables
4. Write paper (8 pages + references)
5. Open-source release

## Key Principles

1. **Token parity is non-negotiable**: Every multi-task claim MUST have a token-controlled single-task comparison
2. **Parameter parity for architecture ablations**: A1–A4 must have comparable trainable parameter counts
3. **Checkpoint every 200 steps**: Kaggle sessions die without warning
4. **Smoke test before every real run**: 50 steps on 100 samples
5. **Auto-batch before every new model**: Never guess VRAM usage
6. **GPT-4 is a reference point, not a baseline**: Frame it honestly

## GPU Strategy

- **Primary**: Kaggle free T4 (16 GB) — handles 80% of project
- **QLoRA 4-bit**: All 7–8B models on T4
- **FP16 LoRA**: Only for 2–3B models on T4
- **Budget**: $0–$80 total (see Budget GPU Guide)
- **Unsloth**: S1/S2 only (verify consistency first)

## Paper Narrative (3 Crisp Contributions)

1. First controlled study of **cross-task transfer vs. data exposure** in medical LLMs (via token-controlled baseline)
2. Parameter-efficient **shared-private adapter design** with empirical justification via architectural ablations (A1–A4)
3. Cost-aware benchmarking under **deployment-realistic constraints** including calibration (ECE) and on-premise feasibility
