# src/results/CLAUDE.md — Results Management System

## Why This Matters

Every experiment produces results that must:
1. **Survive Kaggle session death** — saved to disk after every eval, not just end of training
2. **Be queryable** — "give me all S3b results for Llama-8B across 5 tasks" should be one function call
3. **Feed downstream analysis** — token-controlled baseline (RQ5) needs exact token counts from multi-task runs
4. **Auto-generate paper tables** — `generate_paper_tables.py` reads from this system
5. **Track lineage** — every result links back to its config, checkpoint, and git commit

## Directory Structure

```
results/
├── experiments/                    # One JSON per experiment run
│   ├── phase0_contamination/
│   │   ├── gemma2b_semeval2014t7_zero_shot.json
│   │   ├── llama8b_semeval2017t3_ngram_overlap.json
│   │   └── ...
│   ├── phase1_baselines/
│   │   ├── bert_base_semeval2014t7.json
│   │   ├── biobert_semeval2017t3.json
│   │   └── ...
│   ├── phase2_single_task/
│   │   ├── S1_gemma2b_semeval2014t7_seed42.json
│   │   ├── S1_llama8b_semeval2017t3_seed42.json
│   │   └── ...
│   ├── phase3_multi_task/
│   │   ├── S2_llama8b_all_tasks_seed42.json
│   │   ├── S3a_llama8b_all_tasks_seed42.json
│   │   ├── S3b_llama8b_all_tasks_seed42.json
│   │   ├── S3b_llama8b_token_controlled_semeval2014t7.json  # RQ5
│   │   ├── A1_llama8b_ablation.json                          # Ablation
│   │   ├── A2_llama8b_ablation.json
│   │   ├── A3_llama8b_ablation.json
│   │   ├── A4_llama8b_ablation.json
│   │   ├── S3b_llama8b_mimic_excluded.json                   # MIMIC control
│   │   └── ...
│   ├── phase4_quantization/
│   │   ├── S5_qlora_llama8b_all_tasks.json
│   │   ├── gptq_llama8b_eval.json
│   │   ├── gpt4omini_semeval2014t7_zero_shot.json
│   │   └── ...
│   └── phase5_analysis/
│       ├── probing_shared_adapter_llama8b.json
│       ├── calibration_ece_llama8b_s3b.json
│       ├── error_analysis_llama8b_s3b.json
│       └── ...
├── aggregated/                     # Auto-generated summaries
│   ├── main_results_table.json     # All strategies × models × tasks
│   ├── transfer_matrix.json        # 5×5 transfer heatmap data
│   ├── pareto_frontier.json        # Cost vs performance points
│   ├── token_comparison.json       # RQ5: multi-task vs token-controlled
│   ├── ablation_comparison.json    # A1–A4 results
│   └── negative_transfer.json      # RQ4: which tasks hurt which
├── checkpoints/                    # LoRA adapter weights (10–50 MB each)
│   ├── S1_llama8b_semeval2014t7_best/
│   ├── S3b_llama8b_all_tasks_best/
│   └── ...
└── paper/                          # Auto-generated paper assets
    ├── tables/
    │   ├── tab1_main_results.tex
    │   ├── tab2_efficiency.tex
    │   ├── tab3_ablation.tex
    │   └── tab4_error_analysis.tex
    └── figures/
        ├── fig2_transfer_heatmap.pdf
        ├── fig3_pareto_frontier.pdf
        └── ...
```

## ExperimentResult Schema

Every single experiment run produces exactly one JSON file with this schema:

```python
@dataclass
class ExperimentResult:
    # ── Identity ──
    experiment_id: str          # unique: "{strategy}_{model}_{task}_{seed}_{timestamp}"
    phase: str                  # "phase0", "phase1", ..., "phase5"
    strategy: str               # "S1", "S2", "S3a", "S3b", "S4", "S5", "baseline", "contamination"
    model_name: str             # "gemma2b", "llama8b", etc.
    task: str | list[str]       # "semeval2014t7" or ["all"] for multi-task
    seed: int                   # random seed
    
    # ── Config ──
    config: dict                # full Hydra config snapshot
    git_commit: str             # git hash for reproducibility
    
    # ── Training Stats ──
    training: TrainingStats
        total_steps: int
        total_epochs: float
        tokens_per_task: dict[str, int]   # CRITICAL for RQ5
        total_tokens: int
        best_step: int
        best_epoch: float
        training_time_seconds: float
        peak_vram_gb: float
        samples_per_second: float
        final_train_loss: float
        early_stopped: bool
        early_stop_epoch: int | None
    
    # ── Evaluation Metrics ──
    metrics: dict[str, TaskMetrics]   # one entry per task
        TaskMetrics:
            primary_metric: float         # F1, MAP, etc.
            primary_metric_name: str      # "strict_f1", "map", etc.
            secondary_metrics: dict       # {"relaxed_f1": 0.82, ...}
            confidence_interval: tuple[float, float]  # 95% bootstrap CI
            predictions_path: str | None  # path to full predictions file
    
    # ── Efficiency ──
    efficiency: EfficiencyStats | None
        inference_latency_ms: float       # per-sample, batch=1
        throughput_samples_per_sec: float  # batch=max
        peak_inference_vram_gb: float
        trainable_params: int
        total_params: int
        trainable_ratio: float
        cost_per_1k_inferences: float
    
    # ── Calibration ──
    calibration: CalibrationStats | None
        ece: float                        # Expected Calibration Error
        bin_accuracies: list[float]
        bin_confidences: list[float]
        bin_counts: list[int]
    
    # ── Gradient Conflict (S3a/S3b only) ──
    gradient_conflicts: dict | None
        conflict_frequency: dict[str, dict[str, float]]  # task_pair → frequency
        total_conflicts: int
        total_steps: int
    
    # ── Contamination (Phase 0 only) ──
    contamination: ContaminationResult | None
        zero_shot_f1: float
        zero_shot_threshold: float
        zero_shot_contaminated: bool
        ngram_overlap: dict[int, float]   # n → overlap_ratio
        ngram_contaminated: bool
        min_k_p_value: float
        min_k_contaminated: bool
        overall_contaminated: bool
    
    # ── Metadata ──
    timestamp: str              # ISO 8601
    hostname: str               # "kaggle-xxxxx" or "colab-xxxxx"
    gpu_name: str               # "Tesla T4"
    gpu_vram_gb: float          # 15.1
    notes: str                  # free-form notes
```

## When to Save Results

| Event | What to Save | Where |
|---|---|---|
| After every eval (every 200 steps) | Intermediate metrics + token counts | `results/experiments/phaseN/` (overwrite) |
| End of training (or early stop) | Full ExperimentResult JSON | `results/experiments/phaseN/` (final) |
| Best checkpoint | LoRA adapter weights | `results/checkpoints/` |
| After contamination check | ContaminationResult | `results/experiments/phase0/` |
| After inference benchmark | EfficiencyStats | Appended to existing result JSON |
| After probing | Probing accuracy per probe | `results/experiments/phase5/` |
| After error analysis | Error category counts | `results/experiments/phase5/` |

## ResultsManager API

```python
class ResultsManager:
    """Central API for saving, loading, and querying experiment results."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        
    # ── Save ──
    def save_result(self, result: ExperimentResult) -> Path:
        """Save result to correct phase directory. Returns file path."""
        
    def save_checkpoint(self, model, result: ExperimentResult) -> Path:
        """Save LoRA adapter weights alongside result metadata."""
        
    # ── Load ──
    def load_result(self, experiment_id: str) -> ExperimentResult:
        """Load a specific result by ID."""
        
    def load_all_results(self, phase: str = None, strategy: str = None,
                         model: str = None, task: str = None) -> list[ExperimentResult]:
        """Query results with filters. Returns list sorted by timestamp."""
    
    # ── Query (for downstream analysis) ──
    def get_token_count(self, strategy: str, model: str) -> dict[str, int]:
        """Get tokens_per_task for a specific multi-task run.
        CRITICAL: Used by TokenControlledTrainer to set target tokens."""
        
    def get_best_result(self, strategy: str, model: str, task: str) -> ExperimentResult:
        """Get best result (by primary metric) for a model-task-strategy combo."""
        
    def get_baseline_score(self, model: str, task: str) -> float:
        """Get single-task S1 score for comparison."""
    
    # ── Aggregation (for paper tables) ──
    def build_main_results_table(self) -> pd.DataFrame:
        """Pivot: rows=strategies, columns=tasks, cells=primary_metric.
        Saved to results/aggregated/main_results_table.json"""
        
    def build_transfer_matrix(self) -> np.ndarray:
        """5×5 matrix: [i,j] = perf on task j when also trained on task i.
        Saved to results/aggregated/transfer_matrix.json"""
        
    def build_pareto_frontier(self) -> pd.DataFrame:
        """Columns: model, strategy, quantization, cost_per_1k, avg_score, ece.
        Saved to results/aggregated/pareto_frontier.json"""
        
    def build_token_comparison(self) -> pd.DataFrame:
        """Multi-task score vs token-controlled S1 score per task.
        CRITICAL for RQ5. Saved to results/aggregated/token_comparison.json"""
        
    def build_ablation_table(self) -> pd.DataFrame:
        """A1–A4 results with trainable params.
        Saved to results/aggregated/ablation_comparison.json"""
    
    # ── Paper Generation ──
    def generate_latex_tables(self, output_dir: str = "results/paper/tables"):
        """Auto-generate all LaTeX tables for the paper."""
        
    def generate_figure_data(self, output_dir: str = "results/paper/figures"):
        """Export data needed for matplotlib/seaborn figures."""
    
    # ── Integrity ──
    def validate_all(self) -> dict:
        """Check all results for completeness. Returns missing experiments."""
        
    def check_token_parity(self, multi_task_id: str, token_controlled_ids: list[str]) -> bool:
        """Verify token counts match between MTL and token-controlled runs."""
```

## Integration Points

### 1. Training Loop → ResultsManager
```python
# In trainer.py, after every eval:
result = ExperimentResult(
    experiment_id=f"{strategy}_{model}_{task}_{seed}_{timestamp}",
    phase=current_phase,
    training=TrainingStats(
        tokens_per_task=token_tracker.report(),  # RQ5
        total_tokens=token_tracker.total(),
        ...
    ),
    metrics={task: TaskMetrics(primary_metric=eval_f1, ...)},
    ...
)
results_manager.save_result(result)
```

### 2. Token-Controlled Baseline → ResultsManager
```python
# In token_controlled_trainer.py:
# Step 1: Look up how many tokens the multi-task run used
mtl_tokens = results_manager.get_token_count(strategy="S3b", model="llama8b")
# mtl_tokens = {"semeval2014t7": 45000, "semeval2015t14": 32000, ...}
# total = 150000

# Step 2: Train single-task with that exact total
trainer = TokenControlledTrainer(target_tokens=sum(mtl_tokens.values()))
```

### 3. Paper Generation → ResultsManager
```python
# In generate_paper_tables.py:
rm = ResultsManager("results")
rm.generate_latex_tables()   # → results/paper/tables/tab1_main_results.tex
rm.generate_figure_data()    # → results/paper/figures/
```

## Kaggle Persistence Strategy

```
/kaggle/working/              # Persists as notebook output (20 GB limit)
├── results/
│   ├── experiments/          # JSON results (~1 KB each, negligible)
│   └── checkpoints/          # LoRA adapters (~10-50 MB each)
│       └── keep only best 2 per model-strategy combo

# Between sessions:
# 1. Download results/ folder from notebook output
# 2. Upload as Kaggle Dataset "my-medical-nlp-results"
# 3. In next session, mount dataset and copy to /kaggle/working/results/
```

## Result File Naming Convention

```
{strategy}_{model}_{task}_{variant}_{seed}_{timestamp}.json

Examples:
S1_gemma2b_semeval2014t7_default_seed42_20260215T103000.json
S3b_llama8b_all_tasks_default_seed42_20260301T140000.json
S3b_llama8b_all_tasks_token_controlled_seed42_20260305T090000.json
A1_llama8b_all_tasks_ablation_seed42_20260308T110000.json
contamination_llama8b_semeval2014t7_zero_shot_20260210T080000.json
baseline_biobert_semeval2017t3_default_seed42_20260212T150000.json
```
