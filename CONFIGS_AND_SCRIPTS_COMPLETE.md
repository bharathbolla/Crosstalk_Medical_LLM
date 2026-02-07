# Configs and Scripts Implementation — Complete

**Date**: 2026-02-07
**Status**: ✅ **ALL CONFIGS AND SCRIPTS IMPLEMENTED**

---

## Summary

All Hydra configs, execution scripts, and Jupyter notebook templates have been created.

**Total Files Created**: 30 files
- 6 model configs
- 5 task configs
- 6 strategy configs
- 3 experiment configs (ablation, token-controlled, HP search)
- 1 main config
- 7 execution scripts
- 8 Jupyter notebook templates

---

## Hydra Configs (20 files)

### Model Configs ([configs/model/](configs/model/))
- [x] `gemma2b.yaml` — Google Gemma 2B (2.6B params, FP16)
- [x] `phi3mini.yaml` — Phi-3-mini (3.8B params, FP16)
- [x] `llama3b.yaml` — Llama 3.2 3B (3.2B params, FP16)
- [x] `mistral7b.yaml` — Mistral 7B (7.2B params, QLoRA NF4)
- [x] `llama8b.yaml` — Llama 3.1 8B (8.0B params, QLoRA NF4)
- [x] `biomistral.yaml` — BioMistral 7B (7.2B params, QLoRA NF4)

### Task Configs ([configs/task/](configs/task/))
- [x] `semeval2014t7.yaml` — Clinical Text Analysis (NER, Level 1)
- [x] `semeval2015t14.yaml` — Disorder Identification (Span, Level 1)
- [x] `semeval2016t12.yaml` — Clinical TempEval (Relation, Level 2)
- [x] `semeval2017t3.yaml` — Medical QA (Ranking, Level 2)
- [x] `semeval2021t6.yaml` — Medication Mining (NER+RE, Level 1+2)

### Strategy Configs ([configs/strategy/](configs/strategy/))
- [x] `s1_single.yaml` — Single-task LoRA baseline
- [x] `s2_shared.yaml` — Shared LoRA MTL
- [x] `s3a_flat.yaml` — Flat Shared-Private + PCGrad
- [x] `s3b_hierarchical.yaml` — Hierarchical MTL + PCGrad (PRIMARY NOVELTY)
- [x] `s4_sequential.yaml` — Sequential transfer learning
- [x] `s5_qlora.yaml` — QLoRA 4-bit with best architecture

### Experiment Configs ([configs/experiment/](configs/experiment/))
- [x] `ablation_architecture.yaml` — A1-A4 ablation study with parameter parity
- [x] `token_controlled.yaml` — Token-controlled baseline (RQ5 CRITICAL)
- [x] `hp_search.yaml` — Hyperparameter search space

### Main Config
- [x] `config.yaml` — Main Hydra configuration

---

## Execution Scripts (7 files)

### Main Scripts ([scripts/](scripts/))

1. **[run_experiment.py](scripts/run_experiment.py)** — Main experiment runner
   - Uses Hydra for config management
   - Supports all strategies (S1-S5, S3b)
   - Token tracking enabled by default

2. **[run_contamination_check.py](scripts/run_contamination_check.py)** — Phase 0 contamination detection
   - 3-layer protocol (zero-shot, n-gram, min-k%)
   - All models × all tasks
   - JSON output with flags

3. **[run_baseline.py](scripts/run_baseline.py)** — BERT/BioBERT/ClinicalBERT baselines
   - Standard fine-tuning approach
   - Reproduce published SOTA
   - Phase 1 experiments

4. **[run_gpt4_baseline.py](scripts/run_gpt4_baseline.py)** — GPT-4o-mini API baseline
   - Reference point (not fair comparison)
   - Task-specific prompts
   - Requires OPENAI_API_KEY

5. **[run_inference_bench.py](scripts/run_inference_bench.py)** — Latency/throughput/VRAM profiling
   - Measures deployment metrics
   - Compare quantization levels
   - ECE calibration

6. **[run_probing.py](scripts/run_probing.py)** — Linear probing on trained adapters
   - 4 probe tasks
   - Tests medical knowledge learned
   - Post-training analysis

7. **[generate_paper_tables.py](scripts/generate_paper_tables.py)** — Auto-generate LaTeX tables
   - 5 tables for paper
   - Reads from results JSON
   - Outputs to paper/tables/

---

## Jupyter Notebooks (8 files)

### Analysis Notebooks ([notebooks/](notebooks/))

1. **[01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)**
   - Dataset statistics
   - Sequence length analysis
   - Label schema comparison
   - Vocabulary overlap

2. **[02_contamination_report.ipynb](notebooks/02_contamination_report.ipynb)**
   - Contamination heatmap
   - Per-layer analysis
   - Flagged model-task pairs

3. **[03_transfer_heatmap.ipynb](notebooks/03_transfer_heatmap.ipynb)**
   - 5×5 transfer matrix
   - Negative transfer detection
   - Task similarity correlation (RQ4)

4. **[04_pareto_frontier.ipynb](notebooks/04_pareto_frontier.ipynb)**
   - Model size vs performance
   - Quantization tradeoffs
   - Latency vs performance
   - VRAM vs performance (RQ3)

5. **[05_probing_results.ipynb](notebooks/05_probing_results.ipynb)**
   - Probing task results
   - Compare strategies
   - Medical knowledge analysis

6. **[06_calibration_plots.ipynb](notebooks/06_calibration_plots.ipynb)**
   - Reliability diagrams
   - ECE comparison
   - Confidence calibration

7. **[07_error_analysis.ipynb](notebooks/07_error_analysis.ipynb)**
   - Error categorization (6 categories)
   - Error distribution
   - Common failure modes

8. **[08_token_controlled_analysis.ipynb](notebooks/08_token_controlled_analysis.ipynb)**
   - RQ5 CRITICAL analysis
   - Compare S1 vs S3b vs token-controlled S1
   - Statistical significance testing
   - Core falsifiable claim evaluation

---

## Usage Examples

### Run Main Experiment

```bash
# Single task, single model
python scripts/run_experiment.py model=llama3b task=semeval2017t3 strategy=s1_single

# Multi-task hierarchical (primary novelty)
python scripts/run_experiment.py model=llama3b strategy=s3b_hierarchical task=all

# With experiment config
python scripts/run_experiment.py +experiment=s3b_llama8b

# Token-controlled baseline (RQ5)
python scripts/run_experiment.py +experiment=token_controlled model=llama3b
```

### Run Baselines

```bash
# BERT baseline (Phase 1)
python scripts/run_baseline.py --model bert-base-uncased --task semeval2017t3

# All BERT baselines on all tasks
python scripts/run_baseline.py --all

# BioBERT on all tasks
python scripts/run_baseline.py --model dmis-lab/biobert-base-cased-v1.1 --tasks all
```

### Phase 0: Contamination Check

```bash
# All models and tasks
python scripts/run_contamination_check.py --all

# Specific model
python scripts/run_contamination_check.py --model llama3b --tasks all

# Specific task
python scripts/run_contamination_check.py --model llama3b --task semeval2017t3
```

### Inference Benchmarking

```bash
# Single model
python scripts/run_inference_bench.py --model llama3b --quantization fp16

# Compare quantization levels
python scripts/run_inference_bench.py --model llama8b --quantization fp16,int4

# All models
python scripts/run_inference_bench.py --all
```

### Generate Paper Tables

```bash
python scripts/generate_paper_tables.py --results-dir results/ --output-dir paper/tables/
```

---

## Config Composition with Hydra

Hydra enables flexible config composition:

```bash
# Override any config parameter
python scripts/run_experiment.py \
  model=llama3b \
  task=semeval2017t3 \
  strategy=s1_single \
  strategy.training.learning_rate=5e-5 \
  strategy.training.epochs=10

# Multi-run (sweep over parameters)
python scripts/run_experiment.py -m \
  model=llama3b,phi3mini \
  task=semeval2017t3 \
  strategy=s1_single,s2_shared
```

---

## Execution Order (From CLAUDE.md)

### Phase 0: Setup & Contamination (Weeks 1-3)
1. Apply for PhysioNet access (Day 1)
2. Install dependencies: `pip install -e .`
3. Implement task parsers (fill TODOs)
4. **Run contamination check**: `python scripts/run_contamination_check.py --all`

### Phase 1: Baselines (Week 4)
1. **BERT baselines**: `python scripts/run_baseline.py --all`
2. Verify within 2 F1 points of published SOTA

### Phase 2: Single-Task (Weeks 5-7)
1. **S1 all models**: `python scripts/run_experiment.py model=<model> strategy=s1_single task=all`
2. HP search for top 3 models
3. **Token logging active** (RQ5 critical)

### Phase 3: Multi-Task (Weeks 8-10)
1. **S2**: `python scripts/run_experiment.py strategy=s2_shared`
2. **S3a**: `python scripts/run_experiment.py strategy=s3a_flat`
3. **S3b**: `python scripts/run_experiment.py strategy=s3b_hierarchical`
4. **Token-controlled baseline**: `python scripts/run_experiment.py +experiment=token_controlled`
5. **Ablation study**: Run A1-A4 variants
6. Transfer matrix analysis

### Phase 4: Quantization & Efficiency (Weeks 11-12)
1. **S5 QLoRA**: `python scripts/run_experiment.py strategy=s5_qlora`
2. **Inference benchmarking**: `python scripts/run_inference_bench.py --all`
3. GPT-4 baseline (reference only)

### Phase 5: Analysis & Writing (Weeks 13-16)
1. **Probing**: `python scripts/run_probing.py --all`
2. **Generate tables**: `python scripts/generate_paper_tables.py`
3. Run all analysis notebooks
4. Write paper

---

## Next Steps

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Verify configs load**:
   ```bash
   python scripts/run_experiment.py --help
   ```

3. **Apply for PhysioNet access** (1-2 week approval)

4. **Implement task parsers** (after PhysioNet access granted)

5. **Run first experiment** (smoke test):
   ```bash
   python scripts/run_baseline.py --model bert-base-uncased --task semeval2017t3
   ```

---

**Status**: ✅ All configs and scripts implemented
**Next**: Install dependencies → Implement task parsers → Run Phase 0 contamination check
**Blocking**: PhysioNet access (apply Day 1!)

---

*Last updated: 2026-02-07*
