"""
Validate notebook structure before uploading to Kaggle.
Catches common errors before they waste hours of training time.

Usage: python validate_notebook.py KAGGLE_COMPLETE.ipynb
"""

import sys
import io
import json
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def validate_notebook(notebook_path):
    """Validate notebook structure and content."""

    print("="*60)
    print(f"VALIDATING: {notebook_path}")
    print("="*60)

    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    print(f"\n✓ Total cells: {len(cells)}")

    # Extract all source code
    cell_contents = []
    for i, cell in enumerate(cells):
        source = cell.get('source', [])
        if isinstance(source, list):
            source = ''.join(source)
        cell_contents.append(source)

    full_content = '\n'.join(cell_contents)

    # ============================================
    # CRITICAL CHECKS
    # ============================================

    print("\n" + "="*60)
    print("CRITICAL CHECKS")
    print("="*60)

    # Check 1: Minimum cells
    if len(cells) < 12:
        print(f"❌ Too few cells: {len(cells)} (need >= 12)")
        print("   Missing training execution cells?")
        return False
    print(f"✅ Cell count: {len(cells)} >= 12")

    # Check 2: Must have training call
    if 'trainer.train()' not in full_content:
        print("❌ Missing trainer.train() call!")
        print("   Training will never start!")
        return False
    print("✅ Has trainer.train() call")

    # Check 3: Must have tokenizer loading
    if 'AutoTokenizer' not in full_content:
        print("❌ Missing tokenizer loading!")
        return False
    print("✅ Has tokenizer loading")

    # Check 4: Must have dataset creation
    if 'UniversalMedicalDataset' not in full_content:
        print("❌ Missing UniversalMedicalDataset!")
        return False
    print("✅ Has dataset creation")

    # Check 5: Must have model loading
    if 'AutoModelFor' not in full_content:
        print("❌ Missing model loading!")
        return False
    print("✅ Has model loading")

    # Check 6: Must have smoke test toggle
    if 'SMOKE_TEST' not in full_content:
        print("❌ Missing SMOKE_TEST toggle!")
        return False
    print("✅ Has smoke test toggle")

    # Check 7: Must have CONFIG
    if 'CONFIG' not in full_content:
        print("❌ Missing CONFIG dictionary!")
        return False
    print("✅ Has CONFIG")

    # ============================================
    # ERROR CHECKS
    # ============================================

    print("\n" + "="*60)
    print("ERROR CHECKS")
    print("="*60)

    # Error 1: Double-escaped newlines
    if '\\\\n' in full_content:
        print("❌ Double-escaped newlines found (\\\\n)!")
        print("   This will cause syntax errors!")
        return False
    print("✅ No double-escaped newlines")

    # Error 2: Wrong UniversalMedicalDataset params
    if 'task_type=' in full_content and 'UniversalMedicalDataset' in full_content:
        print("⚠️  WARNING: Found task_type= parameter")
        print("   UniversalMedicalDataset expects task_name, not task_type!")
        print("   This may cause TypeError!")
        # Don't fail, just warn
    else:
        print("✅ No task_type= parameter issues")

    # Error 3: Wrong pickle path
    if 'data/{primary_dataset}_train.pkl' in full_content:
        print("❌ Wrong pickle path: data/{dataset}_train.pkl")
        print("   Correct: data/pickle/{dataset}.pkl")
        return False
    print("✅ Pickle path looks correct")

    # Error 4: Missing is_split_into_words
    if 'tokenizer(tokens' in full_content and 'is_split_into_words=True' not in full_content:
        print("⚠️  WARNING: Tokenizing tokens without is_split_into_words=True")
        print("   This will break NER tasks!")
        # Don't fail, just warn
    else:
        print("✅ Has is_split_into_words=True")

    # ============================================
    # STRUCTURE CHECKS
    # ============================================

    print("\n" + "="*60)
    print("STRUCTURE CHECKS")
    print("="*60)

    # Check cell order
    expected_cells = [
        ('markdown', 'title'),
        ('markdown', 'Cell 1'),
        ('code', 'clone'),
        ('markdown', 'Cell 2'),
        ('code', 'pip install'),
        ('markdown', 'Cell 3'),
        ('code', 'CONFIG'),
        ('markdown', 'Cell 4'),
        ('code', 'SMOKE_TEST'),
    ]

    for i, (expected_type, expected_content) in enumerate(expected_cells[:9]):
        if i >= len(cells):
            print(f"❌ Missing cell {i+1}")
            return False

        cell = cells[i]
        cell_type = cell.get('cell_type', '')
        source = cell.get('source', [])
        if isinstance(source, list):
            source = ''.join(source)

        if cell_type != expected_type:
            print(f"❌ Cell {i+1}: Expected {expected_type}, got {cell_type}")
            return False

    print(f"✅ First 9 cells have correct structure")

    # ============================================
    # CONTENT CHECKS
    # ============================================

    print("\n" + "="*60)
    print("CONTENT CHECKS")
    print("="*60)

    # Check for git clone
    if 'git clone' not in full_content:
        print("❌ Missing git clone command!")
        return False
    print("✅ Has git clone")

    # Check for pip install
    if 'pip install' not in full_content:
        print("❌ Missing pip install!")
        return False
    print("✅ Has pip install")

    # Check for model options
    if 'dmis-lab/biobert' not in full_content:
        print("⚠️  WARNING: No BioBERT model option found")
    else:
        print("✅ Has BioBERT option")

    # Check for evaluation
    if 'trainer.evaluate()' not in full_content:
        print("⚠️  WARNING: No trainer.evaluate() found")
    else:
        print("✅ Has evaluation")

    # ============================================
    # FINAL VERDICT
    # ============================================

    print("\n" + "="*60)
    print("✅ VALIDATION PASSED!")
    print("="*60)
    print("\nNotebook is ready to upload to Kaggle!")
    print("\nNext steps:")
    print("  1. Upload to Kaggle")
    print("  2. Enable GPU (T4)")
    print("  3. Run smoke test (SMOKE_TEST = True)")
    print("  4. Verify F1 > 0.30 in ~2 minutes")
    print("  5. If passed → Set SMOKE_TEST = False for full training")

    return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_notebook.py NOTEBOOK.ipynb")
        print("\nExample:")
        print("  python validate_notebook.py KAGGLE_COMPLETE.ipynb")
        sys.exit(1)

    notebook_path = sys.argv[1]

    if not Path(notebook_path).exists():
        print(f"❌ File not found: {notebook_path}")
        sys.exit(1)

    try:
        success = validate_notebook(notebook_path)
        if success:
            print("\n🎉 All checks passed!")
            sys.exit(0)
        else:
            print("\n❌ Validation failed - fix errors before uploading!")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
