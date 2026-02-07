# GPU Compatibility Assessment - GTX 1080

**Date**: 2026-02-07
**Status**: ⚠️ Local GPU training NOT RECOMMENDED for this project

---

## System Configuration

### Hardware
- **GPU**: NVIDIA GeForce GTX 1080
- **VRAM**: 8 GB GDDR5X
- **CUDA Capability**: 6.1 (Pascal architecture)
- **Driver Version**: 457.51
- **CUDA Version**: 11.1

### Software
- **OS**: Windows
- **Python**: 3.11.0 (main) + 3.14 (additional installation)
- **PyTorch**: 2.10.0+cpu (CPU-only version installed)
- **CUDA Support**: ❌ Not currently available

---

## Installation Issues Encountered

### Problem 1: PyTorch CUDA Installation Failure
- **Issue**: Default pip installation only provides CPU version
- **Attempted**: Installing from CUDA wheel index (`https://download.pytorch.org/whl/cu118`)
- **Result**: Index URLs not accessible or incompatible
- **Impact**: Cannot utilize GPU for training without manual wheel download

### Problem 2: CUDA Version Compatibility
- **System CUDA**: 11.1 (from driver 457.51)
- **PyTorch CUDA**: Requires 11.7+ or 12.1+ for recent versions
- **Gap**: May need driver update for optimal PyTorch compatibility

---

## GTX 1080 Capabilities for This Project

### ✅ What It CAN Handle

**1. Small Models with QLoRA (4-bit)**
- Llama-3.2-1B (4-bit): ~0.5 GB model + ~2 GB training overhead = **2.5 GB total** ✅
- Llama-3.2-3B (4-bit): ~1.5 GB model + ~3 GB training overhead = **4.5 GB total** ✅
- Batch size: 2-4 samples (very limited)

**2. Single-Task Training**
- BC2GM, JNLPBA (NER): Token-level tasks with small batch sizes
- Estimated training time: 4-8 hours per task

**3. Inference Testing**
- Load any quantized model (4-bit/8-bit)
- Run contamination checks
- Test parsers with model tokenization

### ❌ What It CANNOT Handle

**1. Multi-Task Training (Your Core Experiments)**
- **Issue**: Multi-task requires loading multiple task heads simultaneously
- **VRAM needed**: 6-10 GB for 3B models with multiple heads
- **Your GPU**: Only 8 GB total → Out of memory errors likely

**2. Larger Models**
- Llama-3.2-7B: ~3.5 GB (4-bit) + ~5 GB overhead = **8.5 GB** ❌ (exceeds 8 GB)
- Llama-3-8B: Similar constraints

**3. Hierarchical MTL (Strategy S3b)**
- **Requires**: Cross-level attention between Level 1 and Level 2 tasks
- **VRAM impact**: +30-50% over single-task training
- **Your GPU**: Insufficient headroom

**4. Parameter Parity Experiments (A1-A4)**
- **Requires**: Loading multiple architectures for comparison
- **VRAM needed**: 10-12 GB for fair comparisons
- **Your GPU**: Cannot run ablation studies

---

## Kaggle Free T4 Comparison

| Feature | GTX 1080 (Local) | T4 (Kaggle Free) | Winner |
|---------|------------------|-------------------|---------|
| **VRAM** | 8 GB GDDR5X | 16 GB GDDR6 | 🏆 T4 (2x memory) |
| **CUDA Cores** | 2,560 | 2,560 | Tie |
| **Tensor Cores** | ❌ No | ✅ Yes | 🏆 T4 (mixed precision) |
| **FP16 Support** | Limited | Native | 🏆 T4 |
| **Cost** | Free | Free (30h/week) | Tie |
| **Setup Time** | Hours (driver/CUDA issues) | Immediate | 🏆 T4 |
| **Multi-Task Training** | ❌ No | ✅ Yes | 🏆 T4 |
| **Weekly GPU Hours** | Unlimited | 30 hours | Local |

**Verdict**: **Kaggle T4 is superior for 90% of this project**

---

## Recommendations

### 🎯 Primary Strategy: Use Kaggle T4 for All Training

**Reasons**:
1. **VRAM**: 16 GB handles all your experiments (multi-task, hierarchical MTL, ablations)
2. **Tensor Cores**: 2-3x faster training with mixed precision (FP16/BF16)
3. **No Setup Hassle**: Pre-configured with CUDA, PyTorch, transformers
4. **Token-Controlled Baseline (RQ5)**: Can train longer single-task runs without memory issues
5. **Architecture Ablations (A1-A4)**: Sufficient memory for parameter parity experiments

**Kaggle Resource Management**:
- **30 hours/week**: ~4 hours per day
- **Checkpoint every 200 steps**: Survive session disconnects
- **Auto-batch size finder**: Maximize GPU utilization
- **Your project needs**: ~60-80 hours total (fits in 2-3 weeks with planning)

### 🔧 Limited Local GPU Use Cases

**Only use GTX 1080 for**:
1. **Quick smoke tests** (50 steps, 100 samples) before Kaggle runs
2. **Data parsing verification** (load dataset, tokenize, check format)
3. **Contamination checks** (can run in batches overnight)
4. **Probing tasks** (after downloading trained adapters from Kaggle)

**How to enable local GPU**:
```bash
# Option 1: Manual wheel installation (if needed for smoke tests)
# Visit https://pytorch.org/get-started/locally/
# Download CUDA 11.8 wheel for Windows
# pip install <downloaded_wheel>.whl

# Option 2: Skip local GPU training entirely (RECOMMENDED)
# Use CPU for data validation only
# Train everything on Kaggle
```

### 📋 Updated Workflow

**Phase 0: Local (CPU-Only)**
1. ✅ Data parsing (DONE - all 8 parsers working)
2. ⏳ Unit tests with synthetic data (`pytest tests/test_synthetic_data.py`)
3. ⏳ Contamination check (can run on CPU overnight)
4. Upload code to Kaggle notebook

**Phase 1-5: Kaggle T4**
1. All baseline experiments (BERT, BioBERT)
2. All single-task training (S1)
3. All multi-task training (S2, S3a, S3b, S4)
4. Token-controlled baselines (RQ5)
5. Architecture ablations (A1-A4)
6. Quantization experiments (S5, GPTQ, AWQ)

**Post-Training: Local (Download Results)**
1. Download trained adapters from Kaggle
2. Run probing tasks locally
3. Generate figures and tables
4. Write paper

---

## Memory Budget Estimation

### Kaggle T4 (16 GB) - Your Experiments

**Single-Task (S1)**
```
Llama-3.2-3B (BF16):      6 GB
LoRA adapters (r=16):     0.2 GB
Optimizer states:         3 GB
Gradient buffers:         1 GB
Batch (size=8):           2 GB
Safety margin:            2 GB
─────────────────────────────
Total:                   14.2 GB ✅ Fits!
```

**Multi-Task (S3a/S3b)**
```
Llama-3.2-3B (BF16):      6 GB
Shared adapter (r=16):    0.2 GB
5 private adapters (r=8): 0.5 GB
5 task heads:             0.3 GB
Optimizer states:         3.5 GB
Gradient buffers:         1.5 GB
Batch (size=6):           1.5 GB
Safety margin:            1.5 GB
─────────────────────────────
Total:                   15.0 GB ✅ Fits!
```

**Multi-Task with 7B Model (S3b)**
```
Llama-3.2-7B (4-bit QLoRA): 3.5 GB
Adapters + heads:           0.8 GB
Optimizer states:           4.0 GB
Gradient buffers:           2.0 GB
Batch (size=4):             2.0 GB
Safety margin:              2.0 GB
───────────────────────────────
Total:                     14.3 GB ✅ Fits with QLoRA!
```

### GTX 1080 (8 GB) - Limitations

**Single-Task with 3B Model**
```
Llama-3.2-3B (4-bit QLoRA): 1.5 GB
LoRA adapters:              0.2 GB
Optimizer states:           1.5 GB
Gradient buffers:           1.0 GB
Batch (size=2):             1.0 GB
Safety margin:              1.0 GB
───────────────────────────────
Total:                      6.2 GB ✅ Barely fits
```

**Multi-Task (S3a/S3b)**
```
Llama-3.2-3B (4-bit):       1.5 GB
Adapters + 5 heads:         0.7 GB
Optimizer states:           2.5 GB
Gradient buffers:           1.5 GB
Batch (size=2):             1.0 GB
───────────────────────────────
Total:                      7.2 GB ⚠️ Risky, likely OOM
```

---

## Immediate Next Steps

1. **✅ DONE**: All 8 parsers implemented and tested
2. **⏳ NOW**: Run unit tests on CPU
   ```bash
   cd h:\Projects\Research\Cross_Talk_Medical_LLM
   pytest tests/test_synthetic_data.py -v
   ```

3. **⏳ TODAY**: Create Kaggle notebook
   - Upload your code to Kaggle
   - Test data loading on T4
   - Run smoke test (50 steps on BC2GM)

4. **⏳ THIS WEEK**: Run contamination check
   - Option A: Run on local CPU overnight (8-12 hours)
   - Option B: Run on Kaggle T4 (1-2 hours)

5. **⏳ NEXT WEEK**: Start Phase 1 training on Kaggle
   - BERT baseline on BC2GM
   - Verify results match published SOTA

---

## Technical Gotchas to Avoid

### On Kaggle
1. **Session Deaths**: Checkpoint every 200 steps, keep only last 2 checkpoints
2. **Disk Quota**: 20 GB limit → use LoRA adapters only (10-50 MB each)
3. **Auto-Batch Finder**: Always run before real training to avoid OOM

### On Local GPU (If You Enable It)
1. **Driver Update**: May need driver 470+ for CUDA 11.7 support
2. **VRAM Leaks**: Monitor with `nvidia-smi` in loop
3. **Batch Size**: Never exceed 2-4 samples on 8 GB GPU

---

## Conclusion

**Your GTX 1080 is functional but underpowered for this project.**

- ✅ **Good for**: Data validation, smoke tests, contamination checks (CPU)
- ❌ **Bad for**: Multi-task training, hierarchical MTL, large models

**Recommendation**: **Proceed entirely with Kaggle T4** for training (Phases 1-5).

Your local machine is perfect for:
- Code development ✅
- Data parsing ✅ (DONE!)
- Unit testing ✅
- Post-training analysis ✅

**No local GPU training needed** — the 30 hours/week Kaggle T4 budget is sufficient for all experiments.

---

## Decision

**Proceed to Kaggle for full implementation?** → **YES**

Your 8 parsers are ready, unit tests are next, then Kaggle deployment.

---

*Last updated: 2026-02-07*
