# ChemProt Dataset - Temporarily Disabled

**Status**: ⚠️ ChemProt parser disabled due to data loading issues
**Date**: 2026-02-07

---

## Why ChemProt is Disabled

The ChemProt dataset has persistent loading issues with both:
1. **Parquet URL method**: `FileNotFoundError` - URL not accessible
2. **Direct loading method**: `Dataset scripts are no longer supported` error
3. **Trust remote code**: Even with `trust_remote_code=True`, fails on newer datasets library

---

## Impact on Project

### ✅ Minimal Impact

**What we have without ChemProt**:
- ✅ 7 working datasets
- ✅ 48,571 training samples (98% of original 49,591)
- ✅ All task types still covered:
  - NER: BC2GM, JNLPBA (2 tasks)
  - RE: DDI (1 task) ← Still have relation extraction!
  - Classification: GAD, HoC (2 tasks)
  - QA: PubMedQA (1 task)
  - Similarity: BIOSSES (1 task)

**What we lost**:
- ❌ ChemProt: 1,020 training samples (2% of total)
- ❌ Chemical-Protein relation extraction task

---

## Research Design Still Valid

### Hierarchical Multi-Task Learning

```
Level 1 (Entity Recognition):
├── BC2GM (Gene/Protein NER) - 12,574 samples
└── JNLPBA (Bio-entity NER) - 18,607 samples

Level 2 (Higher-level Reasoning):
├── DDI (Drug-Drug Interaction RE) - 571 samples
├── GAD (Gene-Disease Association) - 3,836 samples
├── HoC (Cancer Hallmarks) - 12,119 samples
├── PubMedQA (Medical QA) - 800 samples
└── BIOSSES (Sentence Similarity) - 64 samples
```

**Total**: 48,571 samples across 7 diverse tasks

---

## Files Modified

### Code Files (ChemProt commented out)
1. **src/data/__init__.py**
   - Import commented: `# from .chemprot import ChemProtDataset`
   - Export commented: `# "ChemProtDataset"`

2. **test_parsers.py**
   - Import commented
   - Removed from test suite
   - Updated count: 7 parsers instead of 8

### Download Scripts
1. **download_datasets_parquet.py** (NEW)
   - Uses Parquet URLs
   - Works for all 7 datasets
   - ChemProt excluded

2. **download_diverse_tasks.py** (OLD)
   - Original script with ChemProt
   - Kept for reference

---

## How to Use (Kaggle)

### Download 7 Datasets

Use the Parquet method that works:

```python
from datasets import load_dataset
from pathlib import Path

data_path = Path("data/raw")
data_path.mkdir(parents=True, exist_ok=True)

datasets_config = {
    "bc2gm": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/bc2gm",
    "jnlpba": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/jnlpba",
    # ChemProt excluded
    "ddi": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/ddi_corpus",
    "gad": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/gad",
    "hoc": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/hallmarks_of_cancer",
    "pubmedqa": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/pubmed_qa",
    "biosses": "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/biosses"
}

for name, base_url in datasets_config.items():
    # Build splits
    splits = ["train", "validation", "test"] if name not in ["ddi", "gad"] else ["train", "test"]
    data_files = {split: f"{base_url}/{split}/0000.parquet" for split in splits}

    # Load and save
    dataset = load_dataset("parquet", data_files=data_files)
    dataset.save_to_disk(str(data_path / name))
```

---

## Alternatives (If ChemProt Needed Later)

### Option 1: Manual Download
1. Download ChemProt from original source
2. Parse manually
3. Add to data/raw/

### Option 2: Different Dataset
Use alternative RE dataset:
- BC5CDR (chemicals + diseases)
- GAD already provides gene-disease associations
- DDI covers drug interactions

### Option 3: Wait for Fix
- HuggingFace may fix Parquet URLs
- datasets library may re-enable scripts
- Can add ChemProt later without affecting current work

---

## Recommendation

**Proceed with 7 datasets** and start experiments!

ChemProt provides minimal additional value:
- ✅ Already have 1 RE task (DDI)
- ✅ Already have 48,571 samples
- ✅ All task types covered
- ✅ Hierarchical structure intact

**Focus on getting results, not on one problematic dataset!**

---

## Re-enabling ChemProt

When/if you want to re-enable:

1. Uncomment in `src/data/__init__.py`:
   ```python
   from .chemprot import ChemProtDataset  # Uncomment
   "ChemProtDataset",  # Uncomment in __all__
   ```

2. Uncomment in `test_parsers.py`:
   ```python
   ChemProtDataset,  # Uncomment import
   (ChemProtDataset, "chemprot"),  # Uncomment in parsers list
   ```

3. Download ChemProt data manually or use working method

4. Run `python test_parsers.py` to verify

---

*Last updated: 2026-02-07*
