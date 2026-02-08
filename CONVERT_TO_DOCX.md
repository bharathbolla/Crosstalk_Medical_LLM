# Converting Methodology to DOCX

You have 3 easy options to convert `METHODOLOGY.md` to Word format:

---

## Option 1: Open in Microsoft Word (Easiest)

1. **Open Microsoft Word**
2. **File → Open**
3. Select `METHODOLOGY.md`
4. Word will automatically convert Markdown to formatted document
5. **File → Save As** → Choose `.docx` format

✅ Done! Formatting preserved including tables and code blocks.

---

## Option 2: Use Google Docs

1. **Upload to Google Drive**
   - Drag `METHODOLOGY.md` to Google Drive
2. **Open with Google Docs**
   - Right-click → Open with → Google Docs
3. **Download as Word**
   - File → Download → Microsoft Word (.docx)

✅ Works perfectly, handles all formatting.

---

## Option 3: Use Pandoc (Most Control)

### Install Pandoc:
```bash
# Windows (using chocolatey)
choco install pandoc

# Or download from: https://pandoc.org/installing.html
```

### Convert:
```bash
pandoc METHODOLOGY.md -o METHODOLOGY.docx

# With custom styling:
pandoc METHODOLOGY.md -o METHODOLOGY.docx --reference-doc=template.docx

# With table of contents:
pandoc METHODOLOGY.md -o METHODOLOGY.docx --toc --toc-depth=3
```

### Advanced Options:
```bash
# Professional formatting with TOC and numbering
pandoc METHODOLOGY.md \
  -o METHODOLOGY.docx \
  --toc \
  --toc-depth=3 \
  --number-sections \
  --highlight-style=tango

# For journal submission (double-spaced)
pandoc METHODOLOGY.md \
  -o METHODOLOGY_submission.docx \
  --reference-doc=journal_template.docx
```

---

## What You Get in DOCX:

✅ **Formatted headings** (Heading 1, 2, 3...)
✅ **Tables** (all 10+ tables properly formatted)
✅ **Code blocks** (with syntax highlighting)
✅ **Bullet lists** (properly indented)
✅ **Numbered sections** (optional with Pandoc)
✅ **Table of Contents** (optional with Pandoc)

---

## For Your Research Paper:

### Method 1: Copy Sections to Paper

Open `METHODOLOGY.docx` and copy specific sections:
- Section 2 → Research Questions
- Section 3 → Dataset Description
- Section 5 → Training Procedures
- Section 6 → Experimental Configuration
- Section 7 → Evaluation Metrics

### Method 2: Use as Supplementary Material

Keep `METHODOLOGY.docx` as supplementary material:
- Main paper: Summarize key points (2-3 pages)
- Supplementary: Full methodology (this document)

---

## Quick Reference:

| File | Format | Use For |
|------|--------|---------|
| `METHODOLOGY.md` | Markdown | Version control, easy editing |
| `METHODOLOGY.docx` | Word | Journal submission, collaboration |
| `METHODOLOGY.pdf` | PDF | Final submission, archival |

---

## Tips for Paper Writing:

### For Methods Section (Main Paper):

Extract these parts:
- Section 3.1: Dataset Selection (Table 1)
- Section 5.3: Token-Controlled Baseline (YOUR KEY CONTRIBUTION!)
- Section 6.1: Hyperparameters (Table)
- Section 7.1: Evaluation Metrics

**Target**: 2-3 pages for Methods in main paper

### For Supplementary Material:

Include full `METHODOLOGY.docx` with:
- All 14 sections
- Complete hyperparameter justifications
- Detailed token tracking implementation
- Full experimental procedure

---

## Example: Insert into Paper Template

```bash
# If you have a journal template (e.g., ACL, EMNLP):

# 1. Convert methodology to DOCX
pandoc METHODOLOGY.md -o METHODOLOGY.docx

# 2. Open both files in Word
# - paper_template.docx
# - METHODOLOGY.docx

# 3. Copy-paste relevant sections
# - Keep template formatting
# - Adjust heading levels as needed
# - Insert tables where appropriate

# 4. Update references
# - Cite datasets
# - Cite baseline papers
```

---

## Troubleshooting:

### Tables not formatting well?
→ Use Pandoc with `--reference-doc` to control table style

### Code blocks losing formatting?
→ In Word: Select code → Font: Courier New, Size: 10

### Need double-spacing for submission?
→ Word: Home → Paragraph → Line spacing → 2.0

### Want numbered sections?
→ Pandoc: Add `--number-sections` flag

---

## Ready to Use!

Just open `METHODOLOGY.md` in Word or run:
```bash
pandoc METHODOLOGY.md -o METHODOLOGY.docx --toc
```

You'll have a professional Word document ready for your paper! 📄
