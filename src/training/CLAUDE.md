# src/training/CLAUDE.md — Training Instructions

## Training Loop Requirements

### Non-Negotiable Components
1. **Token counter**: Log cumulative tokens per task every step
2. **VRAM monitor**: Check for leaks every 50 steps
3. **Checkpoint manager**: Save every 200 steps (Kaggle safety)
4. **Smoke test**: Run 50 steps on 100 samples before any real training
5. **Quick eval**: Evaluate on 500 dev samples every 200 steps
6. **Early stopping**: Patience=3 epochs on dev metric

### trainer.py — MultiTaskTrainer

```python
class MultiTaskTrainer:
    """
    Extends HuggingFace Trainer for multi-task learning.
    
    Key features:
    - Dynamic task sampling (proportional/uniform/temperature)
    - Per-task loss tracking and weighting
    - Token counting for RQ5 control
    - PCGrad integration for S3a/S3b
    - Checkpoint resume for Kaggle sessions
    
    Training loop pseudocode:
    
    for step in range(total_steps):
        task = sample_task(strategy)
        batch = get_batch(task)
        
        # Token counting (RQ5)
        token_tracker.update(task, batch)
        
        if strategy.uses_pcgrad:
            # Compute per-task losses, then project gradients
            pcgrad_optimizer.step(task_losses)
        else:
            loss = model(batch, task)
            loss.backward()
            optimizer.step()
        
        # Monitoring
        if step % 50 == 0:
            vram_monitor.check(step, loss)
        if step % 200 == 0:
            quick_eval(model, dev_loaders)
            checkpoint_manager.save(...)
    """
```

### pcgrad.py — Gradient Conflict Resolution

```python
class PCGradOptimizer:
    """
    Projected Conflicting Gradients (Yu et al., 2020).
    
    When Task A gradient conflicts with Task B gradient (negative dot product),
    project A's gradient onto the normal plane of B's gradient.
    
    Used ONLY for S3a and S3b strategies.
    Adds ~15-20% training overhead.
    
    Log conflict frequency:
    - Record how often conflicts occur between each task pair
    - This informs the negative transfer analysis (RQ4)
    """
```

### loss.py — Multi-Task Loss

```python
class UncertaintyWeightedLoss(nn.Module):
    """
    Learns task weights automatically via homoscedastic uncertainty
    (Kendall et al., 2018).
    
    loss = sum_i (1/(2*sigma_i^2)) * L_i + log(sigma_i)
    
    where sigma_i is a learned parameter per task.
    
    Alternative (for ablation): equal weighting
    loss = sum_i L_i / num_tasks
    """
```

### callbacks.py — Monitoring

```python
class TrainingCallbacks:
    """
    Callback suite for budget-constrained research:
    
    1. VRAMCallback: Monitors VRAM every 50 steps, alerts on leaks
    2. LossCallback: Alerts on NaN/explosion, logs per-task loss curves
    3. SpeedCallback: Tracks samples/sec, estimates ETA
    4. TokenCallback: Logs cumulative tokens per task (RQ5)
    5. GradientConflictCallback: Logs PCGrad conflict frequency (RQ4)
    """
```

## Token-Controlled Baseline Procedure (RQ5)

This is the **most important experimental control** in the entire project.

### Step 1: Train multi-task model (S2, S3a, or S3b)
- Log total tokens seen across all tasks
- Example: S3b sees 150K tokens total (30K per task × 5 tasks)

### Step 2: Train token-controlled single-task model
- For each task, train S1 model with **exactly 150K tokens** of that task
- This means oversampling: if task has 9K sentences, repeat ~17 times
- Use the same learning rate schedule, just more epochs

### Step 3: Compare
- If multi-task > token-controlled single-task → **genuine transfer**
- If multi-task ≈ token-controlled single-task → gains are from data exposure only
- If multi-task < token-controlled single-task → **negative transfer**

```python
class TokenControlledTrainer(MultiTaskTrainer):
    """
    Trains single-task model with exact token budget from multi-task run.
    
    Args:
        target_tokens: int  # total tokens to train on (from multi-task log)
        task: str           # which task to oversample
    """
    
    def should_stop(self) -> bool:
        return self.token_tracker.total() >= self.target_tokens
```

## MIMIC Exclusion Control

To test whether shared adapter benefits come from seeing similar MIMIC data across tasks (rather than genuine cross-task transfer):

1. Train S3a/S3b **excluding all MIMIC-based tasks** from shared adapter
2. Only use 2017-T3 (public) and 2021-T6 (public) for shared adapter
3. Then fine-tune private adapters on MIMIC tasks
4. Compare with full shared adapter (all 5 tasks)

If performance drops significantly → MIMIC data overlap was driving the gains, not cross-task transfer.

## Kaggle Session Management

### Before Each Session
```python
# 1. Check for existing checkpoint
ckpt = checkpoint_manager.load_latest()
if ckpt:
    print(f"Resuming from step {ckpt.step}")
else:
    print("Starting fresh")

# 2. Auto-detect batch size
optimal_bs = find_optimal_batch_size(model, ...)

# 3. Smoke test
assert smoke_test(model, train_loader), "Smoke test failed!"

# 4. Start training from checkpoint
trainer.train(resume_from=ckpt)
```

### Kaggle-Specific Settings
```python
# Save results to /kaggle/working/ (persists as notebook output)
SAVE_DIR = Path("/kaggle/working/results")

# Save adapters (10-50 MB) not full models (3-16 GB)
SAVE_FULL_MODEL = False

# Aggressive checkpointing
CHECKPOINT_EVERY = 200  # steps

# Session time limit
MAX_SESSION_HOURS = 11.5  # leave 30 min buffer before Kaggle kills
```
