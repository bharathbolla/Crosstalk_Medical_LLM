# Dataset Download Status & Solutions

**Date**: 2026-02-07
**Status**: ⚠️ **4/5 datasets require manual download**

---

## Current Situation

| Dataset | HuggingFace Download | Status | Issue |
|---------|---------------------|--------|-------|
| **PubMedQA** | ✅ Working | Downloaded | pubmed_qa dataset |
| **BC5CDR** | ❌ Failed | Not downloaded | Deprecated loading script |
| **NCBI-Disease** | ❌ Failed | Not downloaded | Deprecated loading script |
| **DDI** | ❌ Failed | Not downloaded | No modern format |
| **GAD** | ❌ Failed | Not downloaded | No modern format |

---

## Problem Explanation

**HuggingFace has deprecated dataset loading scripts** (as of late 2024). Medical/biomedical datasets have been slow to migrate to the new Parquet-only format. This affects most older NLP datasets including BC5CDR, NCBI-Disease, DDI, and GAD.

Error message:
```
Dataset scripts are no longer supported, but found [dataset_name].py
```

**Only PubMedQA works** because it's been migrated to the modern format.

---

## Solution 1: Use Kaggle Datasets (RECOMMENDED)

**Advantages:**
- ✅ Pre-processed and ready to use
- ✅ Already uploaded by community
- ✅ Easy to add as input to your Kaggle notebook
- ✅ No need to download locally first

### Step 1: Find Datasets on Kaggle

Search for these datasets on https://www.kaggle.com/datasets:

1. **BC5CDR**: Search "BC5CDR biomedical"
   - Example: https://www.kaggle.com/datasets/[user]/bc5cdr

2. **NCBI-Disease**: Search "NCBI disease corpus"
   - Example: https://www.kaggle.com/datasets/[user]/ncbi-disease

3. **DDI**: Search "drug drug interaction corpus"
   - Example: https://www.kaggle.com/datasets/[user]/ddi-corpus

4. **GAD**: Search "gene disease association"
   - Example: https://www.kaggle.com/datasets/[user]/gad-biobert

### Step 2: Add to Your Kaggle Notebook

1. Click "Add Data" in your Kaggle notebook
2. Search for and select the datasets
3. They'll be available in `/kaggle/input/[dataset-name]/`

### Step 3: Copy to Your Project

```python
# In Kaggle notebook
!cp -r /kaggle/input/bc5cdr-dataset/data ./data/raw/bc5cdr
!cp -r /kaggle/input/ncbi-disease-dataset/data ./data/raw/ncbi_disease
!cp -r /kaggle/input/ddi-dataset/data ./data/raw/ddi
!cp -r /kaggle/input/gad-dataset/data ./data/raw/gad
```

---

## Solution 2: Upload Your Own Datasets to Kaggle

If datasets aren't available on Kaggle, you can:

### Step 1: Download Locally from Original Sources

Use one of these methods:

#### Option A: Use Provided Manual Download Script

```bash
# This script downloads from original sources
python scripts/download_datasets_manual.py --all
```

**Note**: The manual script URLs may need updating. Check these sources:

| Dataset | Official Source | Alternative |
|---------|----------------|-------------|
| BC5CDR | [NCBI FTP](https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/) | [GitHub](https://github.com/JHnlp/BioCreative-V-CDR-Corpus) |
| NCBI-Disease | [NCBI BioNLP](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/) | [GitHub Mirror](https://github.com/spyysalo/ncbi-disease) |
| DDI | [GitHub](https://github.com/isegura/DDICorpus) | - |
| GAD | [Archived at NIH](https://geneticassociationdb.nih.gov/) | [BioBERT preprocessed](https://github.com/dmis-lab/biobert) |

#### Option B: Manual Browser Download

1. Visit the links above
2. Download ZIP/tar.gz files
3. Extract to `data/raw/[dataset_name]/`

### Step 2: Create Kaggle Dataset

1. **Zip your data**:
```bash
cd data/raw
tar -czf bc5cdr.tar.gz bc5cdr/
tar -czf ncbi_disease.tar.gz ncbi_disease/
tar -czf ddi.tar.gz ddi/
tar -czf gad.tar.gz gad/
```

2. **Upload to Kaggle**:
   - Go to https://www.kaggle.com/datasets
   - Click "New Dataset"
   - Upload the .tar.gz files
   - Title: "Medical NLP Datasets (BC5CDR, NCBI, DDI, GAD)"
   - Privacy: Private (recommended)
   - Click "Create"

3. **Use in your notebooks**:
   - Add as input dataset to any notebook
   - Extract and use

---

## Solution 3: Download Directly in Kaggle

Download datasets fresh in each Kaggle session (has internet access):

```python
# In Kaggle notebook

# 1. Install required tools
!pip install -q requests beautifulsoup4

# 2. Download from original sources
# BC5CDR from NCBI FTP
!wget https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/CDR_Data.zip
!unzip CDR_Data.zip -d data/raw/bc5cdr/

# NCBI-Disease (manual process - use Kaggle datasets instead)
# DDI from GitHub
!wget https://github.com/isegura/DDICorpus/archive/refs/heads/master.zip
!unzip master.zip -d data/raw/ddi/

# GAD - download TSV files
!wget https://raw.githubusercontent.com/dmis-lab/biobert/master/dataset/GAD/train.tsv -O data/raw/gad/train.tsv
# (Note: If 404, URLs changed - check GitHub repo)
```

**Pros**: Always get latest data
**Cons**: Takes time at start of each session

---

## Solution 4: Alternative Datasets

If the above datasets are too difficult to obtain, consider these alternatives:

### Working HuggingFace Alternatives:

| Original | Alternative | HuggingFace Name | Task Type |
|----------|-------------|------------------|-----------|
| BC5CDR | BioRED | (search for) | NER + Relations |
| NCBI-Disease | (None modern) | - | Disease NER |
| DDI | ChemProt | bigbio/chemprot | Chemical-protein interactions |
| GAD | (Use local) | - | Gene-disease |
| PubMedQA | ✅ **WORKING** | pubmed_qa | QA |

**Trade-off**: Different tasks, may require different parsers

---

## Recommended Approach for Your Project

### For Local Development (Windows):

1. ✅ PubMedQA already downloaded
2. **Skip the others** for now - implement PubMedQA parser first
3. Test full pipeline with 1 dataset before downloading others

### For Kaggle Deployment:

1. ✅ Push code to GitHub (without datasets)
2. Search for pre-uploaded datasets on Kaggle
3. Add them as inputs to your notebook
4. If not found, download locally → upload as Kaggle dataset

---

## Next Steps (Priority Order)

### Immediate (Today):

1. **Implement PubMedQA parser** - you have the data!
   ```bash
   # Edit: src/data/pubmedqa.py
   # Test with: python -c "from src.data.pubmedqa import PubMedQAParser"
   ```

2. **Test full pipeline** with PubMedQA:
   ```bash
   python scripts/run_baseline.py --model bert-base-uncased --task pubmedqa
   ```

### Short-term (This Week):

3. **Search Kaggle for datasets**:
   - BC5CDR
   - NCBI-Disease
   - DDI
   - GAD

4. **Upload to Kaggle** if not found:
   - Download from original sources
   - Create Kaggle dataset
   - Make private/public as needed

### Before Experiments (Next Week):

5. **Implement remaining 4 parsers**:
   - `src/data/bc5cdr.py`
   - `src/data/ncbi_disease.py`
   - `src/data/ddi.py`
   - `src/data/gad.py`

6. **Test with sample data** (even if incomplete):
   - Create minimal test files
   - Verify parser logic works
   - Fix bugs before real data

---

## Dataset Parser Implementation Order

**Recommended order** (easiest to hardest):

1. **PubMedQA** (QA) - Downloaded, JSON format, clean structure
2. **GAD** (Relation) - TSV format, simple binary classification
3. **NCBI-Disease** (NER) - XML format, single entity type
4. **BC5CDR** (NER+Relation) - PubTator format, 2 entity types + relations
5. **DDI** (Relation) - XML format, multiple relation types

---

## Summary

**What works now:**
- ✅ PubMedQA downloaded via HuggingFace
- ✅ Scripts ready for manual download
- ✅ Kaggle deployment guide ready

**What you need to do:**
1. Implement PubMedQA parser (highest priority)
2. Search Kaggle for pre-uploaded datasets
3. If not found, download and upload to Kaggle yourself
4. Implement remaining parsers

**Estimated time:**
- PubMedQA parser: 2-4 hours
- Finding/uploading Kaggle datasets: 1-2 hours
- Remaining 4 parsers: 8-12 hours total

---

## Sources

The dataset sources mentioned in this document:

- [BC5CDR NCBI FTP](https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/)
- [BC5CDR GitHub Mirror](https://github.com/JHnlp/BioCreative-V-CDR-Corpus)
- [NCBI-Disease Official](https://www.ncbi.nlm.nih.gov/research/bionlp/Data/disease/)
- [NCBI-Disease GitHub](https://github.com/spyysalo/ncbi-disease)
- [DDI Corpus GitHub](https://github.com/isegura/DDICorpus)
- [GAD Database](https://geneticassociationdb.nih.gov/)
- [BioBERT Datasets](https://github.com/dmis-lab/biobert)
- [PubMedQA GitHub](https://github.com/pubmedqa/pubmedqa)

---

*Last updated: 2026-02-07*
