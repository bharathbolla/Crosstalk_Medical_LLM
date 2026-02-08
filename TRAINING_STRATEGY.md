# Training Strategy: Why Early Stopping Over Fixed Epochs

## ❌ Problem with Fixed 3 Epochs

The original configuration used **3 epochs** which is problematic for research:

### Issues:

1. **Arbitrary choice**: Why 3 and not 2, 5, or 10?
   - No theoretical justification
   - Just borrowed from other papers
   - Different datasets need different amounts

2. **Dataset-dependent convergence**:
   - BC2GM (12K samples): Might overfit after epoch 2
   - ChemProt (40K samples): Might need 5+ epochs to converge
   - PubMedQA (1K samples): Could overfit after 1 epoch

3. **Unfair comparisons**:
   - Small datasets overtrained → inflated validation F1, poor test F1
   - Large datasets undertrained → poor performance across the board
   - Can't compare results fairly across datasets

4. **Not publication-ready**:
   - Reviewers will ask: "Why 3 epochs specifically?"
   - Hard to defend without empirical evidence

---

## ✅ Solution: Early Stopping + Max 10 Epochs

### Updated Configuration:

```python
CONFIG = {
    "num_epochs": 10,  # Maximum epochs
    "use_early_stopping": True,
    "early_stopping_patience": 3,  # Stop if no improvement for 3 evals
    "early_stopping_threshold": 0.0001,  # Minimum meaningful improvement
    "eval_steps": 250,  # Evaluate frequently
}
```

### How It Works:

1. **Set max epochs high** (10): Allows model to fully converge
2. **Evaluate frequently** (every 250 steps): Monitor validation F1
3. **Stop when plateaus** (patience=3): No improvement for 3 consecutive evals
4. **Load best checkpoint**: Return model from best eval, not final

### Example Training Progression:

```
Epoch 1.0 → Eval F1: 0.7234
Epoch 1.3 → Eval F1: 0.7512 ✓ (improvement)
Epoch 1.6 → Eval F1: 0.7689 ✓ (improvement)
Epoch 2.0 → Eval F1: 0.7834 ✓ (improvement) [BEST]
Epoch 2.3 → Eval F1: 0.7821 (no improvement, patience=1)
Epoch 2.6 → Eval F1: 0.7798 (no improvement, patience=2)
Epoch 3.0 → Eval F1: 0.7811 (no improvement, patience=3)
→ STOP! Best was at epoch 2.0 with F1=0.7834
```

**Actual training**: Stopped at epoch 3.0 (not 10!)
**Best model**: From epoch 2.0
**Result**: Optimal performance without overfitting

---

## 📊 Why This Is Better for Research

### 1. Scientifically Rigorous

- **Justified stopping**: Based on validation performance, not arbitrary number
- **Reproducible**: Same stopping criteria across all experiments
- **Transparent**: Report actual epochs trained in results

### 2. Dataset-Adaptive

Each dataset finds its own optimal point:
- Small datasets: Might stop at epoch 2-3 (prevents overfit)
- Large datasets: Might train to epoch 6-8 (allows convergence)
- All get their best possible performance

### 3. Fair Comparisons

- All models trained to their optimal point
- Single-task vs multi-task: Both get equal opportunity to converge
- Token-controlled baseline (RQ5): Can match token count precisely

### 4. Publication-Ready

**Methods section**:
> "We trained all models for a maximum of 10 epochs with early stopping
> (patience=3, threshold=0.0001) based on validation F1 score. We used
> the checkpoint with the best validation performance for final evaluation."

**Results table**:
```
Dataset    | Model      | Epochs* | Val F1  | Test F1
-----------|------------|---------|---------|--------
BC2GM      | BioBERT    | 2.6     | 0.7834  | 0.7621
JNLPBA     | BioBERT    | 4.2     | 0.8123  | 0.7956
ChemProt   | BioBERT    | 6.8     | 0.7234  | 0.7102
*Stopped early based on validation performance
```

**Reviewers will appreciate**:
- Not cherry-picking epochs per dataset
- Consistent methodology
- Prevents overfitting

---

## 🎯 When to Use Different Settings

### Standard Experiments (Recommended):
```python
"num_epochs": 10,
"early_stopping_patience": 3,
"eval_steps": 250,
```
→ **Use for**: All baselines, multi-task, comparisons

### Quick Tests:
```python
"num_epochs": 3,
"use_early_stopping": False,
"max_samples_per_dataset": 100,
```
→ **Use for**: Verifying code works before full run

### Large Datasets:
```python
"num_epochs": 15,
"early_stopping_patience": 5,
"eval_steps": 500,
```
→ **Use for**: Datasets with >50K samples

### Tiny Datasets:
```python
"num_epochs": 20,
"early_stopping_patience": 2,
"eval_steps": 100,
```
→ **Use for**: PubMedQA (1K samples) - needs more epochs per eval

---

## 🔬 Expected Behavior by Dataset

Based on medical NLP literature:

| Dataset   | Size  | Expected Epochs | Typical Best |
|-----------|-------|-----------------|--------------|
| BC2GM     | 12K   | 2-4             | Epoch 2-3    |
| JNLPBA    | 24K   | 3-5             | Epoch 3-4    |
| ChemProt  | 40K   | 4-8             | Epoch 5-7    |
| DDI       | 8K    | 2-3             | Epoch 2      |
| GAD       | 5K    | 2-3             | Epoch 2-3    |
| HoC       | 4K    | 2-4             | Epoch 2-3    |
| PubMedQA  | 1K    | 5-15            | Epoch 8-12   |
| BIOSSES   | 100   | 10-20           | Epoch 15+    |

**Note**: Multi-task may converge faster (shared learning) or slower (task interference)

---

## 💡 Pro Tips

### 1. Always Log Actual Epochs
```python
# In results JSON:
"actual_epochs_trained": train_result.metrics['epoch'],
"best_checkpoint_epoch": trainer.state.best_model_checkpoint,
```

### 2. Plot Learning Curves
After experiments, analyze:
- Did it stop too early? → Increase patience
- Did it train too long? → Decrease patience
- Oscillating validation? → Reduce learning rate

### 3. Report in Paper
**Table caption**:
> "Results using early stopping (patience=3, max 10 epochs).
> All models evaluated on checkpoint with best validation F1."

**Supplementary**:
- Include learning curves for each dataset
- Show early stopping saved X% training time

### 4. For RQ5 (Token-Controlled)
```python
# Multi-task trains to epoch 4.2 → 5.2M tokens
# Single-task needs to match:
CONFIG = {
    "target_tokens": 5200000,
    "use_early_stopping": False,  # Disable! Need exact token match
    "num_epochs": 20,  # High max, stop when tokens reached
}
```

---

## 📚 References from Literature

**BioBERT (2019)**:
> "We fine-tuned for 3 epochs on biomedical NER tasks"
→ Fixed 3 epochs, no justification

**PubMedBERT (2020)**:
> "Training for 2-4 epochs depending on task size"
→ Manual tuning per dataset (not scalable)

**Clinical-BERT (2019)**:
> "Early stopping with patience of 3 epochs"
→ **This is what we're doing!** (Best practice)

**Your Paper (2026)**:
> "We employed early stopping with patience of 3 evaluations to prevent
> overfitting while allowing sufficient training for convergence."
→ **Scientifically rigorous! ✅**

---

## Summary

| Approach | Pros | Cons | Use For |
|----------|------|------|---------|
| **Fixed 3 epochs** | Simple, fast | Arbitrary, unfair | Quick tests only |
| **Early stopping** | Optimal, fair, rigorous | Slightly slower | **All real experiments** |
| **Manual tuning** | Can optimize per dataset | Not scalable, biased | Never (too subjective) |

**Recommendation**: **Always use early stopping for research experiments!**

---

## Quick Reference

**In Cell 3 of notebook**:
```python
CONFIG = {
    # For real experiments:
    "num_epochs": 10,
    "use_early_stopping": True,
    "early_stopping_patience": 3,

    # For quick tests (5 min):
    "num_epochs": 1,
    "use_early_stopping": False,
    "max_samples_per_dataset": 100,
}
```

**That's it! Let the model tell you when it's done, not an arbitrary number.**
