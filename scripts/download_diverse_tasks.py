"""Download DIVERSE medical NLP task datasets from HuggingFace.

This script downloads datasets across multiple task types:
- Named Entity Recognition (NER)
- Relation Extraction (RE)
- Document Classification
- Question Answering (QA)
- Sentence Similarity

Usage:
    python scripts/download_diverse_tasks.py --all
    python scripts/download_diverse_tasks.py --task-type ner
    python scripts/download_diverse_tasks.py --task-type relation
"""

import sys
import io
import argparse
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("[!] 'datasets' library not installed")


# ============================================================================
# LEVEL 1 TASKS: Named Entity Recognition (NER)
# ============================================================================

def download_bc2gm(data_dir: Path):
    """Download BC2GM (Gene/Protein NER) - Level 1."""
    print("\n" + "=" * 60)
    print("LEVEL 1 (NER): BC2GM - Gene/Protein Recognition")
    print("=" * 60)

    try:
        base_url = "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/bc2gm"
        dataset = load_dataset('parquet', data_files={
            'train': f'{base_url}/train/0000.parquet',
            'validation': f'{base_url}/validation/0000.parquet',
            'test': f'{base_url}/test/0000.parquet'
        })

        output_dir = data_dir / "bc2gm"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] BC2GM downloaded")
        print(f"  Task: Named Entity Recognition (Gene/Protein)")
        print(f"  Size: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] BC2GM failed: {str(e)[:150]}")
        return False


def download_jnlpba(data_dir: Path):
    """Download JNLPBA (Bio-entity NER) - Level 1."""
    print("\n" + "=" * 60)
    print("LEVEL 1 (NER): JNLPBA - Bio-entity Recognition")
    print("=" * 60)

    try:
        base_url = "https://huggingface.co/datasets/bigbio/blurb/resolve/refs%2Fconvert%2Fparquet/jnlpba"
        dataset = load_dataset('parquet', data_files={
            'train': f'{base_url}/train/0000.parquet',
            'validation': f'{base_url}/validation/0000.parquet',
            'test': f'{base_url}/test/0000.parquet'
        })

        output_dir = data_dir / "jnlpba"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] JNLPBA downloaded")
        print(f"  Task: Named Entity Recognition (DNA, RNA, Protein, Cell Line, Cell Type)")
        print(f"  Size: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] JNLPBA failed: {str(e)[:150]}")
        return False


# ============================================================================
# LEVEL 2 TASKS: Relation Extraction (RE)
# ============================================================================

def download_chemprot(data_dir: Path):
    """Download ChemProt (Chemical-Protein Relation) - Level 2."""
    print("\n" + "=" * 60)
    print("LEVEL 2 (RE): ChemProt - Chemical-Protein Relations")
    print("=" * 60)

    try:
        base_url = "https://huggingface.co/datasets/bigbio/chemprot/resolve/refs%2Fconvert%2Fparquet/chemprot_bigbio_kb"
        dataset = load_dataset('parquet', data_files={
            'train': f'{base_url}/train/0000.parquet',
            'validation': f'{base_url}/validation/0000.parquet',
            'test': f'{base_url}/test/0000.parquet'
        })

        output_dir = data_dir / "chemprot"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] ChemProt downloaded")
        print(f"  Task: Relation Extraction (Chemical-Protein interactions)")
        print(f"  Size: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] ChemProt failed: {str(e)[:150]}")
        return False


def download_ddi(data_dir: Path):
    """Download DDI (Drug-Drug Interaction) - Level 2."""
    print("\n" + "=" * 60)
    print("LEVEL 2 (RE): DDI - Drug-Drug Interactions")
    print("=" * 60)

    try:
        base_url = "https://huggingface.co/datasets/bigbio/ddi_corpus/resolve/refs%2Fconvert%2Fparquet/ddi_corpus_bigbio_kb"
        dataset = load_dataset('parquet', data_files={
            'train': f'{base_url}/train/0000.parquet',
            'test': f'{base_url}/test/0000.parquet'
        })

        output_dir = data_dir / "ddi"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] DDI downloaded")
        print(f"  Task: Relation Extraction (Drug-Drug interactions)")
        print(f"  Size: {len(dataset['train'])} train, {len(dataset['test'])} test")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] DDI failed: {str(e)[:150]}")
        return False


def download_gad(data_dir: Path):
    """Download GAD (Gene-Disease Association) - Level 2."""
    print("\n" + "=" * 60)
    print("LEVEL 2 (RE): GAD - Gene-Disease Associations")
    print("=" * 60)

    try:
        base_url = "https://huggingface.co/datasets/bigbio/gad/resolve/refs%2Fconvert%2Fparquet/gad_fold0_bigbio_text"
        dataset = load_dataset('parquet', data_files={
            'train': f'{base_url}/train/0000.parquet',
            'test': f'{base_url}/test/0000.parquet'
        })

        output_dir = data_dir / "gad"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] GAD downloaded")
        print(f"  Task: Relation Extraction (Gene-Disease associations)")
        print(f"  Size: {len(dataset['train'])} train, {len(dataset['test'])} test")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] GAD failed: {str(e)[:150]}")
        return False


# ============================================================================
# LEVEL 2 TASKS: Document Classification
# ============================================================================

def download_hoc(data_dir: Path):
    """Download HoC (Hallmarks of Cancer) - Level 2."""
    print("\n" + "=" * 60)
    print("LEVEL 2 (Classification): HoC - Hallmarks of Cancer")
    print("=" * 60)

    try:
        base_url = "https://huggingface.co/datasets/bigbio/hallmarks_of_cancer/resolve/refs%2Fconvert%2Fparquet/hallmarks_of_cancer_bigbio_text"
        dataset = load_dataset('parquet', data_files={
            'train': f'{base_url}/train/0000.parquet',
            'validation': f'{base_url}/validation/0000.parquet',
            'test': f'{base_url}/test/0000.parquet'
        })

        output_dir = data_dir / "hoc"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] HoC downloaded")
        print(f"  Task: Document Classification (Cancer hallmarks)")
        print(f"  Size: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] HoC failed: {str(e)[:150]}")
        return False


# ============================================================================
# LEVEL 2 TASKS: Question Answering (QA)
# ============================================================================

def download_pubmedqa(data_dir: Path):
    """Download PubMedQA (Medical QA) - Level 2."""
    print("\n" + "=" * 60)
    print("LEVEL 2 (QA): PubMedQA - Medical Question Answering")
    print("=" * 60)

    try:
        dataset = load_dataset("pubmed_qa", "pqa_labeled")

        output_dir = data_dir / "pubmedqa"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] PubMedQA downloaded")
        print(f"  Task: Question Answering (yes/no/maybe)")
        print(f"  Size: {len(dataset['train'])} train")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] PubMedQA failed: {str(e)[:150]}")
        return False


def download_bioasq(data_dir: Path):
    """Download BioASQ (Biomedical QA) - Level 2."""
    print("\n" + "=" * 60)
    print("LEVEL 2 (QA): BioASQ - Biomedical Question Answering")
    print("=" * 60)

    try:
        # BioASQ has multiple tasks, try the main one
        base_url = "https://huggingface.co/datasets/bigbio/bioasq_task_b/resolve/refs%2Fconvert%2Fparquet/bioasq_4b_source"
        dataset = load_dataset('parquet', data_files={
            'train': f'{base_url}/train/0000.parquet',
            'test': f'{base_url}/test/0000.parquet'
        })

        output_dir = data_dir / "bioasq"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] BioASQ downloaded")
        print(f"  Task: Question Answering (biomedical)")
        print(f"  Size: {len(dataset['train'])} train, {len(dataset['test'])} test")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] BioASQ failed: {str(e)[:150]}")
        return False


# ============================================================================
# LEVEL 2 TASKS: Sentence Similarity
# ============================================================================

def download_biosses(data_dir: Path):
    """Download BIOSSES (Sentence Similarity) - Level 2."""
    print("\n" + "=" * 60)
    print("LEVEL 2 (Similarity): BIOSSES - Sentence Similarity")
    print("=" * 60)

    try:
        base_url = "https://huggingface.co/datasets/bigbio/biosses/resolve/refs%2Fconvert%2Fparquet/biosses_bigbio_pairs"
        dataset = load_dataset('parquet', data_files={
            'train': f'{base_url}/train/0000.parquet',
            'validation': f'{base_url}/validation/0000.parquet',
            'test': f'{base_url}/test/0000.parquet'
        })

        output_dir = data_dir / "biosses"
        dataset.save_to_disk(str(output_dir))

        print(f"[OK] BIOSSES downloaded")
        print(f"  Task: Sentence Similarity")
        print(f"  Size: {len(dataset['train'])} train, {len(dataset['validation'])} val, {len(dataset['test'])} test")
        print(f"  Location: {output_dir}")
        return True
    except Exception as e:
        print(f"[X] BIOSSES failed: {str(e)[:150]}")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download diverse medical NLP datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--task-type", type=str,
                        choices=["ner", "relation", "classification", "qa", "similarity"],
                        help="Download specific task type")
    parser.add_argument("--dataset", type=str,
                        choices=["bc2gm", "jnlpba", "chemprot", "ddi", "gad", "hoc", "pubmedqa", "bioasq", "biosses"],
                        help="Download specific dataset")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Output directory")

    args = parser.parse_args()

    if not HAS_DATASETS:
        print("\nPlease install the datasets library:")
        print("  pip install datasets")
        return 1

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DIVERSE MEDICAL NLP TASK DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {data_dir.absolute()}\n")
    print("Task Types:")
    print("  Level 1: NER (Named Entity Recognition)")
    print("  Level 2: RE (Relation Extraction), Classification, QA, Similarity")
    print()

    results = {}

    if args.all:
        # Level 1: NER
        results['bc2gm'] = download_bc2gm(data_dir)
        results['jnlpba'] = download_jnlpba(data_dir)
        # Level 2: Relation Extraction
        results['chemprot'] = download_chemprot(data_dir)
        results['ddi'] = download_ddi(data_dir)
        results['gad'] = download_gad(data_dir)
        # Level 2: Classification
        results['hoc'] = download_hoc(data_dir)
        # Level 2: QA
        results['pubmedqa'] = download_pubmedqa(data_dir)
        results['bioasq'] = download_bioasq(data_dir)
        # Level 2: Similarity
        results['biosses'] = download_biosses(data_dir)

    elif args.task_type:
        if args.task_type == "ner":
            results['bc2gm'] = download_bc2gm(data_dir)
            results['jnlpba'] = download_jnlpba(data_dir)
        elif args.task_type == "relation":
            results['chemprot'] = download_chemprot(data_dir)
            results['ddi'] = download_ddi(data_dir)
            results['gad'] = download_gad(data_dir)
        elif args.task_type == "classification":
            results['hoc'] = download_hoc(data_dir)
        elif args.task_type == "qa":
            results['pubmedqa'] = download_pubmedqa(data_dir)
            results['bioasq'] = download_bioasq(data_dir)
        elif args.task_type == "similarity":
            results['biosses'] = download_biosses(data_dir)

    elif args.dataset:
        dataset_map = {
            'bc2gm': download_bc2gm,
            'jnlpba': download_jnlpba,
            'chemprot': download_chemprot,
            'ddi': download_ddi,
            'gad': download_gad,
            'hoc': download_hoc,
            'pubmedqa': download_pubmedqa,
            'bioasq': download_bioasq,
            'biosses': download_biosses
        }
        results[args.dataset] = dataset_map[args.dataset](data_dir)

    else:
        print("Please specify --all, --task-type, or --dataset")
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    print(f"\n[OK] Downloaded: {success_count}/{total_count} datasets")

    # Group by task type
    ner_tasks = {k: v for k, v in results.items() if k in ['bc2gm', 'jnlpba']}
    re_tasks = {k: v for k, v in results.items() if k in ['chemprot', 'ddi', 'gad']}
    class_tasks = {k: v for k, v in results.items() if k in ['hoc']}
    qa_tasks = {k: v for k, v in results.items() if k in ['pubmedqa', 'bioasq']}
    sim_tasks = {k: v for k, v in results.items() if k in ['biosses']}

    if ner_tasks:
        print(f"\nLevel 1 - NER ({sum(ner_tasks.values())}/{len(ner_tasks)}):")
        for name, success in ner_tasks.items():
            status = "[OK]" if success else "[X]"
            print(f"  {status} {name}")

    if re_tasks:
        print(f"\nLevel 2 - Relation Extraction ({sum(re_tasks.values())}/{len(re_tasks)}):")
        for name, success in re_tasks.items():
            status = "[OK]" if success else "[X]"
            print(f"  {status} {name}")

    if class_tasks:
        print(f"\nLevel 2 - Classification ({sum(class_tasks.values())}/{len(class_tasks)}):")
        for name, success in class_tasks.items():
            status = "[OK]" if success else "[X]"
            print(f"  {status} {name}")

    if qa_tasks:
        print(f"\nLevel 2 - Question Answering ({sum(qa_tasks.values())}/{len(qa_tasks)}):")
        for name, success in qa_tasks.items():
            status = "[OK]" if success else "[X]"
            print(f"  {status} {name}")

    if sim_tasks:
        print(f"\nLevel 2 - Similarity ({sum(sim_tasks.values())}/{len(sim_tasks)}):")
        for name, success in sim_tasks.items():
            status = "[OK]" if success else "[X]"
            print(f"  {status} {name}")

    if success_count == total_count:
        print("\n[SUCCESS] All datasets downloaded successfully!")
        print("\nHierarchical MTL Structure:")
        print("  Level 1 (NER): bc2gm, jnlpba")
        print("  Level 2 (RE): chemprot, ddi, gad")
        print("  Level 2 (Class): hoc")
        print("  Level 2 (QA): pubmedqa, bioasq")
        print("  Level 2 (Sim): biosses")
        print("\nNext steps:")
        print("  1. Implement parsers in src/data/")
        print("  2. Update configs/strategy/s3b_hierarchical.yaml")
        print("  3. Start experiments!")
    else:
        print(f"\n[!] {total_count - success_count} dataset(s) failed")
        print("  Check errors above for details")

    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
