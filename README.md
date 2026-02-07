# Medical Cross-Task Knowledge Transfer with Small Language Models

**Research Question**: Can multi-task fine-tuning enable small language models (2B–8B parameters) to achieve substantial performance gains on medical NLP tasks through genuine cross-task knowledge transfer, rather than mere data exposure?

## Core Falsifiable Claim

> If multi-task models outperform single-task models **even when total training tokens are equalized**, this constitutes evidence of genuine cross-task knowledge transfer rather than data exposure alone.

## Project Overview

This project studies cross-task knowledge transfer in medical NLP using 5 SemEval shared tasks:
- **SemEval-2014 Task 7**: Clinical Text Analysis (disorder span detection)
- **SemEval-2015 Task 14**: Disorder identification with discontiguous entities
- **SemEval-2016 Task 12**: Clinical TempEval (temporal relations)
- **SemEval-2017 Task 3**: Community Question Answering (medical QA)
- **SemEval-2021 Task 6**: Medication Event Classification (NER + relation extraction)

### Key Innovations

1. **Token-Controlled Baseline (RQ5)**: First study to explicitly separate gains from cross-task transfer vs. increased data exposure
2. **Parameter-Efficient Architecture**: Shared-private LoRA adapters with architectural ablations
3. **Deployment-Realistic Constraints**: All experiments on free Kaggle T4 GPU (16 GB VRAM), with calibration and cost analysis

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/medical-cross-task-transfer.git
cd medical-cross-task-transfer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For faster training (optional, requires CUDA)
pip install -e ".[fast]"
```

### Configuration

```bash
# Set up wandb (optional)
wandb login

# Create .env file
echo "WANDB_PROJECT=medical-cross-task" > .env
echo "KAGGLE_USERNAME=your_username" >> .env
echo "KAGGLE_KEY=your_key" >> .env
```

### Running Experiments

```bash
# Phase 0: Contamination check
python scripts/run_contamination_check.py

# Phase 1: Single-task baseline (S1)
python scripts/run_experiment.py strategy=S1 model=phi3_mini task=semeval2014t7

# Phase 2: Multi-task (S2)
python scripts/run_experiment.py strategy=S2 model=phi3_mini

# Phase 3: Token-controlled baseline
python scripts/run_experiment.py strategy=S1_token_controlled model=phi3_mini

# Generate paper tables
python scripts/generate_paper_tables.py
```

## Project Structure

```
medical-cross-task-transfer/
├── CLAUDE.md                   # Master orchestration file
├── configs/                    # Hydra configuration files
├── src/                        # Source code
│   ├── data/                   # Data parsers and loaders
│   ├── models/                 # Model architectures
│   ├── training/               # Training loop and optimizers
│   ├── evaluation/             # Metrics and contamination checks
│   ├── quantization/           # QLoRA, GPTQ, AWQ
│   ├── utils/                  # Auto-batch, checkpointing, monitoring
│   └── results/                # Results management and aggregation
├── scripts/                    # Entry points for experiments
├── notebooks/                  # Analysis and visualization
└── results/                    # Saved experiment results (auto-generated)
```

## Experiment Strategies

- **S1**: Single-task LoRA fine-tuning (baseline)
- **S1_token_controlled**: Single-task with matched token count (critical control)
- **S2**: Shared LoRA multi-task
- **S3a**: Flat Shared-Private LoRA + PCGrad
- **S3b**: Hierarchical MTL + PCGrad
- **S4**: Sequential transfer learning
- **S5**: QLoRA 4-bit quantization

## Models Evaluated

- Phi-3-mini-4k (3.8B)
- Phi-3-small-8k (7B)
- Gemma-2-2b
- Gemma-2-9b
- Llama-3.2-3B
- Qwen2.5-7B

## Research Questions

1. **RQ1**: Does multi-task fine-tuning improve per-task performance after controlling for contamination?
2. **RQ2**: Can shared adapters enable out-of-distribution task generalization?
3. **RQ3**: What is the Pareto frontier of model size, quantization, and performance?
4. **RQ4**: When does negative transfer occur, and can it be predicted?
5. **RQ5 (Critical)**: Do multi-task gains persist under token-controlled conditions?

## GPU Requirements

- **Primary**: Free Kaggle T4 (16 GB) — handles 80% of experiments
- **QLoRA 4-bit**: All 7–8B models
- **FP16 LoRA**: 2–3B models only
- **Budget**: $0–$80 total across all phases

## Citation

```bibtex
@article{medical-cross-task-2026,
  title={Cross-Task Knowledge Transfer with Small Language Models: Evidence from Medical NLP},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License - see LICENSE file for details

## Contributing

This is a research project. For questions or collaboration inquiries, please open an issue.

## Acknowledgments

- SemEval shared task organizers
- PhysioNet for MIMIC-III access
- Kaggle for free GPU resources
