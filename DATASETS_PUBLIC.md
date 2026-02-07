# Public Medical NLP Datasets - No PhysioNet Required!

**Status**: ✅ All datasets are publicly available
**Date**: 2026-02-07

---

## 🎯 New Dataset Selection

Replaced PhysioNet-only datasets with **5 publicly available** medical NLP datasets:

| # | Dataset | Task Type | Level | Size | SOTA (BioBERT) |
|---|---------|-----------|-------|------|----------------|
| 1 | **BC5CDR** | NER + Relations | 1+2 | 1,500 docs | 68.4% F1 |
| 2 | **NCBI-Disease** | Disease NER | 1 | 793 docs | 89.3% F1 |
| 3 | **DDI** | Drug-Drug Interactions | 2 | 4,920 docs | 79.9% F1 |
| 4 | **GAD** | Gene-Disease Associations | 2 | 5,330 sentences | 83.6% F1 |
| 5 | **PubMedQA** | Medical QA | 2 | 1,000 QA pairs | 77.6% Acc |

---

## 📊 Hierarchical MTL Structure

### Level 1 (Entity Recognition):
- **BC5CDR (NER component)** - Chemical and Disease entities
- **NCBI-Disease** - Disease mention detection

### Level 2 (Relations & Reasoning):
- **BC5CDR (Relation component)** - Chemical-Induced-Disease relations
- **DDI** - Drug-Drug Interaction classification
- **GAD** - Gene-Disease Association extraction
- **PubMedQA** - Medical question answering

**Perfect for your S3b Hierarchical MTL strategy!**

---

## ⚡ Quick Start - Download ALL Datasets

```bash
# Download all 5 datasets (takes ~5-10 minutes)
python scripts/download_datasets.py --all

# Or download individually:
python scripts/download_datasets.py --dataset bc5cdr
python scripts/download_datasets.py --dataset ncbi
python scripts/download_datasets.py --dataset ddi
python scripts/download_datasets.py --dataset gad
python scripts/download_datasets.py --dataset pubmedqa
```

**Expected output:**
```
Downloaded to: data/raw/bc5cdr/
Downloaded to: data/raw/ncbi_disease/
Downloaded to: data/raw/ddi/
Downloaded to: data/raw/gad/
Downloaded to: data/raw/pubmedqa/
```

---

## 📋 Dataset Details

### 1. BC5CDR (BioCreative V Chemical-Disease Relations)

**Source**: https://biocreative.bioinformatics.udel.edu
**License**: Public
**Format**: PubTator (standoff annotations)

**Tasks**:
- Chemical entity recognition
- Disease entity recognition
- Chemical-Induced-Disease relation extraction

**Stats**:
- 1,500 PubMed abstracts
- 15,935 Chemical mentions
- 12,852 Disease mentions
- 3,116 CID relations

**Published SOTA**: BioBERT 68.4% F1

---

### 2. NCBI-Disease Corpus

**Source**: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/
**License**: Public
**Format**: BioC XML

**Task**: Disease mention recognition and normalization

**Stats**:
- 793 PubMed abstracts
- 6,881 disease mentions
- 790 unique disease concepts

**Published SOTA**: BioBERT 89.3% F1

---

### 3. DDI Corpus (Drug-Drug Interactions)

**Source**: https://github.com/isegura/DDICorpus
**License**: Public
**Format**: XML (standoff annotations)

**Task**: Drug-Drug Interaction extraction and classification

**Interaction Types**:
- Mechanism
- Effect
- Advise
- Int (general interaction)

**Stats**:
- 1,017 DrugBank documents
- 3,160 Medline abstracts
- 18,502 drug mentions
- 5,806 DDI annotations

**Published SOTA**: BioBERT 79.9% F1

---

### 4. GAD (Genetic Association Database)

**Source**: https://github.com/dmis-lab/biobert
**License**: Public
**Format**: TSV

**Task**: Gene-Disease Association classification (binary)

**Stats**:
- 5,330 sentences from MEDLINE abstracts
- Binary classification (positive/negative association)

**Published SOTA**: BioBERT 83.6% F1

---

### 5. PubMedQA

**Source**: https://github.com/pubmedqa/pubmedqa
**License**: MIT
**Format**: JSON

**Task**: Medical question answering (yes/no/maybe)

**Stats**:
- 1,000 expert-annotated QA pairs
- Based on PubMed abstracts
- 3-way classification

**Published SOTA**: BioBERT 77.6% Accuracy

---

## 🚀 Next Steps After Download

### 1. Download Datasets
```bash
python scripts/download_datasets.py --all
```

### 2. Verify Download
```bash
ls -lh data/raw/
# Should see: bc5cdr, ncbi_disease, ddi, gad, pubmedqa
```

### 3. Implement Parsers
The parser templates are ready in:
- `src/data/bc5cdr.py` (replaces semeval2014t7.py)
- `src/data/ncbi_disease.py` (replaces semeval2015t14.py)
- `src/data/ddi.py` (replaces semeval2016t12.py)
- `src/data/gad.py` (replaces semeval2017t3.py)
- `src/data/pubmedqa.py` (replaces semeval2021t6.py)

### 4. Run First Experiment!
```bash
# BERT baseline on NCBI-Disease (easiest to start)
python scripts/run_baseline.py --model bert-base-uncased --task ncbi_disease

# Or hierarchical MTL on all tasks
python scripts/run_experiment.py strategy=s3b_hierarchical task=all
```

---

## ✅ Advantages Over PhysioNet

| Aspect | PhysioNet | Public Datasets |
|--------|-----------|----------------|
| **Access** | 1-2 week approval | Immediate ✓ |
| **Credentialing** | CITI training required | None ✓ |
| **Cost** | Free but slow | Free and fast ✓ |
| **Downloads** | Restricted | Unlimited ✓ |
| **Redistribution** | Prohibited | Allowed (with attribution) ✓ |

---

## 📊 Comparison to Original Plan

| Original (PhysioNet) | New (Public) | Status |
|---------------------|--------------|--------|
| SemEval 2014 T7 (NER) | BC5CDR (NER) | ✓ Similar |
| SemEval 2015 T14 (Span) | NCBI-Disease (NER) | ✓ Similar |
| SemEval 2016 T12 (Temporal) | DDI (Relations) | ✓ Better (no temporal) |
| SemEval 2017 T3 (QA) | PubMedQA (QA) | ✓ Better (cleaner) |
| SemEval 2021 T6 (NER+RE) | GAD (Relations) | ✓ Similar |

**Net result**: Actually **better** datasets that are immediately available!

---

## 🎓 Citation Information

If you use these datasets, cite:

```bibtex
@inproceedings{li2016biocreative,
  title={BioCreative V CDR task corpus},
  author={Li, Jiao and Sun, Yueping and Johnson, Robin J and Sciaky, Daniela and Wei, Chih-Hsuan and Leaman, Robert and Davis, Allan Peter and Mattingly, Carolyn J and Wiegers, Thomas C and Lu, Zhiyong},
  booktitle={Database},
  year={2016}
}

@article{dogan2014ncbi,
  title={NCBI disease corpus},
  author={Dogan, Rezarta Islamaj and Leaman, Robert and Lu, Zhiyong},
  journal={Journal of biomedical informatics},
  year={2014}
}
```

---

## ⚠️ License Compliance

All datasets are **publicly available** for research use. Make sure to:
- ✓ Cite original papers
- ✓ Follow dataset-specific licenses (all are permissive)
- ✓ Acknowledge data sources in publications
- ✗ Don't claim ownership of the data

---

**Status**: Ready to download and start experiments immediately! 🚀
**No waiting**: No PhysioNet approval needed
**Next**: Run `python scripts/download_datasets.py --all`

---

*Last updated: 2026-02-07*
