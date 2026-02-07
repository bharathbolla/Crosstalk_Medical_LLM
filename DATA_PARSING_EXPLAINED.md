# Data Parsing Explained - Complete Guide

**Date**: 2026-02-07

---

## What is Data Parsing?

**Data Parsing** = Converting datasets from their **download format** into your **training format**.

Think of it like translating between languages:
- **Download format**: Each dataset speaks its own "language"
- **Training format**: Your model only understands ONE "language" (UnifiedSample)
- **Parser**: The translator between them

---

## The Problem: 8 Different Formats

You downloaded 8 datasets. Each has a **different structure**:

### Example 1: BC2GM (NER - Gene/Protein)

**What you downloaded**:
```python
{
  "id": "0",
  "tokens": ["Immunohistochemical", "staining", "was", "positive", "for", "S", "-", "100"],
  "ner_tags": [0, 0, 0, 0, 0, 1, 2, 2]
  #           O  O  O  O  O  B  I  I  (B=Gene start, I=Gene continue, O=Not entity)
}
```

**What your model needs**:
```python
UnifiedSample(
    text="Immunohistochemical staining was positive for S - 100",
    tokens=["Immunohistochemical", "staining", "was", "positive", "for", "S", "-", "100"],
    entities=[
        Entity(text="S - 100", entity_type="Gene", start_token=5, end_token=8)
    ],
    task_type="ner",
    task_name="bc2gm"
)
```

### Example 2: ChemProt (Relation Extraction)

**What you downloaded**:
```python
{
  "passages": [{"text": "Methotrexate inhibits tumor necrosis factor."}],
  "entities": [
    {"id": "e1", "type": "CHEMICAL", "text": ["methotrexate"], "offsets": [[0, 12]]},
    {"id": "e2", "type": "GENE-N", "text": ["tumor necrosis factor"], "offsets": [[23, 44]]}
  ],
  "relations": [
    {"type": "INHIBITOR", "arg1_id": "e1", "arg2_id": "e2"}
  ]
}
```

**What your model needs**:
```python
UnifiedSample(
    text="Methotrexate inhibits tumor necrosis factor.",
    tokens=["Methotrexate", "inhibits", "tumor", "necrosis", "factor", "."],
    entities=[
        Entity(text="methotrexate", entity_type="Chemical", ...),
        Entity(text="tumor necrosis factor", entity_type="Gene", ...)
    ],
    relations=[
        Relation(head="e1", tail="e2", type="INHIBITOR")
    ],
    task_type="relation_extraction",
    task_name="chemprot"
)
```

### Example 3: PubMedQA (Question Answering)

**What you downloaded**:
```python
{
  "question": "Do mitochondria play a role in programmed cell death?",
  "context": {"contexts": ["Mitochondria are involved in apoptosis..."]},
  "final_decision": "yes"
}
```

**What your model needs**:
```python
UnifiedSample(
    text="[Q] Do mitochondria play a role in programmed cell death? [A] Mitochondria are involved in apoptosis...",
    tokens=["[Q]", "Do", "mitochondria", "play", "a", "role", ...],
    entities=[],
    relations=[],
    label="yes",  # The answer!
    task_type="question_answering",
    task_name="pubmedqa"
)
```

---

## The Solution: Write 8 Parsers

Each parser is a Python class that converts one dataset's format → UnifiedSample.

---

## Complete Working Example: BC2GM Parser

### Step 1: Look at Raw Data

```bash
# What BC2GM looks like
{
  "tokens": ["S", "-", "100", "in", "all", "cases"],
  "ner_tags": [1, 2, 2, 0, 0, 0]
  # 0 = O (outside entity)
  # 1 = B-Gene (beginning of gene)
  # 2 = I-Gene (inside gene)
}
```

### Step 2: Write Parser

```python
# File: src/data/bc2gm.py

from pathlib import Path
from typing import List
from datasets import load_from_disk

from .base import UnifiedSample, Entity


class BC2GMParser:
    """Parse BC2GM dataset (Gene/Protein NER) to UnifiedSample format."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.dataset = None

    def load(self):
        """Load dataset from disk."""
        self.dataset = load_from_disk(str(self.data_dir / "bc2gm"))
        return self

    def parse_split(self, split: str = "train") -> List[UnifiedSample]:
        """Parse one split (train/validation/test) to UnifiedSample format."""
        if self.dataset is None:
            self.load()

        samples = []
        for raw_sample in self.dataset[split]:
            # Convert this sample
            sample = self._parse_sample(raw_sample)
            samples.append(sample)

        return samples

    def _parse_sample(self, raw_sample: dict) -> UnifiedSample:
        """Convert one BC2GM sample to UnifiedSample."""

        # Extract fields
        tokens = raw_sample['tokens']
        ner_tags = raw_sample['ner_tags']

        # Convert BIO tags to Entity objects
        entities = self._bio_to_entities(tokens, ner_tags)

        # Create UnifiedSample
        return UnifiedSample(
            text=" ".join(tokens),
            tokens=tokens,
            entities=entities,
            relations=[],  # BC2GM has no relations
            label=None,    # BC2GM has no label
            task_type="ner",
            task_name="bc2gm"
        )

    def _bio_to_entities(self, tokens: List[str], bio_tags: List[int]) -> List[Entity]:
        """Convert BIO tags (0=O, 1=B, 2=I) to Entity objects."""
        entities = []
        current_entity = None

        for i, (token, tag) in enumerate(zip(tokens, bio_tags)):
            if tag == 1:  # B-Gene (beginning of new gene)
                # Save previous entity if exists
                if current_entity:
                    entities.append(self._create_entity(current_entity, tokens))

                # Start new entity
                current_entity = {
                    'start': i,
                    'tokens': [token],
                    'type': 'Gene'
                }

            elif tag == 2:  # I-Gene (continuation)
                if current_entity:
                    current_entity['tokens'].append(token)
                else:
                    # I tag without B tag - treat as new entity
                    current_entity = {
                        'start': i,
                        'tokens': [token],
                        'type': 'Gene'
                    }

            else:  # tag == 0 (O - outside)
                if current_entity:
                    entities.append(self._create_entity(current_entity, tokens))
                    current_entity = None

        # Don't forget last entity
        if current_entity:
            entities.append(self._create_entity(current_entity, tokens))

        return entities

    def _create_entity(self, entity_dict: dict, all_tokens: List[str]) -> Entity:
        """Create Entity object from entity dictionary."""
        start_token = entity_dict['start']
        end_token = start_token + len(entity_dict['tokens'])

        # Calculate character offsets
        text_before = " ".join(all_tokens[:start_token])
        start_char = len(text_before) + (1 if text_before else 0)
        entity_text = " ".join(entity_dict['tokens'])
        end_char = start_char + len(entity_text)

        return Entity(
            text=entity_text,
            entity_type=entity_dict['type'],
            start_char=start_char,
            end_char=end_char,
            start_token=start_token,
            end_token=end_token
        )


# Usage
if __name__ == "__main__":
    parser = BC2GMParser()
    train_samples = parser.parse_split("train")

    print(f"Parsed {len(train_samples)} training samples")
    print(f"\nExample sample:")
    print(f"  Text: {train_samples[0].text[:100]}...")
    print(f"  Entities: {train_samples[0].entities}")
```

### Step 3: Test It

```python
# Test the parser
from src.data.bc2gm import BC2GMParser

parser = BC2GMParser()
samples = parser.parse_split("train")

print(f"Total samples: {len(samples)}")
print(f"\nFirst sample:")
print(f"  Text: {samples[0].text}")
print(f"  Tokens: {samples[0].tokens[:10]}")
print(f"  Entities: {samples[0].entities[:3]}")
print(f"  Task: {samples[0].task_name}")
```

**Output**:
```
Total samples: 12574

First sample:
  Text: Immunohistochemical staining was positive for S - 100 in all 9 cases...
  Tokens: ['Immunohistochemical', 'staining', 'was', 'positive', 'for', 'S', '-', '100', 'in', 'all']
  Entities: [Entity(text='S - 100', type='Gene', start_token=5, end_token=8),
             Entity(text='HMB - 45', type='Gene', start_token=16, end_token=19),
             Entity(text='cytokeratin', type='Gene', start_token=31, end_token=32)]
  Task: bc2gm
```

---

## What Each Parser Does (Summary)

| Dataset | Input Format | Parser Converts | Output |
|---------|--------------|-----------------|--------|
| **BC2GM** | BIO tags | `ner_tags` → `entities` | UnifiedSample with entities |
| **JNLPBA** | BIO tags (5 types) | `ner_tags` → `entities` (DNA, RNA, etc) | UnifiedSample with entities |
| **ChemProt** | Entity + relation dicts | `entities` + `relations` → objects | UnifiedSample with entities + relations |
| **DDI** | Entity + relation dicts | `entities` + `relations` → objects | UnifiedSample with entities + relations |
| **GAD** | Text + label | `text` + `label` → classification | UnifiedSample with label |
| **HoC** | Text + label | `text` + `label` → classification | UnifiedSample with label |
| **PubMedQA** | Question + context + answer | Combine all → QA format | UnifiedSample with label |
| **BIOSSES** | Sentence pair + score | `sentence1` + `sentence2` + `score` | UnifiedSample with similarity |

---

## Why This Matters

### Without Parsers (Your code breaks):
```python
# Training loop - DOESN'T WORK
for sample in dataset:
    # What fields exist? Depends on dataset!
    tokens = sample['tokens']  # ❌ ChemProt doesn't have 'tokens'!
    labels = sample['label']   # ❌ BC2GM doesn't have 'label'!
```

### With Parsers (Your code works):
```python
# Training loop - WORKS FOR ALL DATASETS
for sample in dataset:
    # Every sample is UnifiedSample - same structure!
    tokens = sample.tokens      # ✓ Always exists
    entities = sample.entities  # ✓ Always exists (empty list if no entities)
    relations = sample.relations # ✓ Always exists (empty list if no relations)
    label = sample.label        # ✓ Always exists (None if not classification)
```

---

## Next Steps

### 1. Understand UnifiedSample Format

Read `src/data/base.py` to see the exact definition:

```python
@dataclass
class UnifiedSample:
    text: str                      # Full text
    tokens: List[str]              # Tokenized words
    entities: List[Entity]         # NER entities
    relations: List[Relation]      # RE relations
    label: Optional[str]           # Classification/QA label
    task_type: str                 # "ner", "relation_extraction", "qa", etc
    task_name: str                 # "bc2gm", "chemprot", etc
    # ... other fields
```

### 2. Implement 8 Parsers

**Priority order** (easiest to hardest):

1. **BC2GM** (done above!) - Copy and modify for JNLPBA
2. **PubMedQA** (1 hour) - Simple: question + context + answer
3. **GAD** (1 hour) - Simple: text + binary label
4. **HoC** (1 hour) - Similar to GAD: text + multi-label
5. **BIOSSES** (1 hour) - Sentence pairs + similarity score
6. **DDI** (2 hours) - Entities + relations
7. **ChemProt** (2 hours) - Complex: entities + relations
8. **JNLPBA** (1.5 hours) - Like BC2GM but 5 entity types

### 3. Test Each Parser

```python
# For each parser
parser = BC2GMParser()  # Or ChemProtParser, PubMedQAParser, etc
samples = parser.parse_split("train")

# Verify
assert len(samples) > 0, "No samples parsed!"
assert all(isinstance(s, UnifiedSample) for s in samples), "Wrong type!"
print(f"✓ {parser.__class__.__name__} works!")
```

### 4. Register Parsers

Add to `src/data/__init__.py`:

```python
from .bc2gm import BC2GMParser
from .jnlpba import JNLPBAParser
from .chemprot import ChemProtParser
# ... etc

PARSERS = {
    'bc2gm': BC2GMParser,
    'jnlpba': JNLPBAParser,
    'chemprot': ChemProtParser,
    # ... etc
}

def get_parser(task_name: str):
    """Get parser for a task."""
    return PARSERS[task_name]()
```

---

## Estimated Time

- **BC2GM**: 1.5 hours (example done!)
- **JNLPBA**: 1.5 hours (similar to BC2GM, 5 entity types)
- **PubMedQA**: 1 hour (simple)
- **GAD**: 1 hour (simple)
- **HoC**: 1 hour (similar to GAD)
- **BIOSSES**: 1 hour (sentence pairs)
- **DDI**: 2 hours (entities + relations)
- **ChemProt**: 2 hours (complex entities + relations)

**Total**: ~10-12 hours of focused work

---

## Templates Provided

You have templates in `src/data/` already! Check:
- `src/data/base.py` - UnifiedSample definition
- Look at any existing `semeval*.py` files for structure

Just copy the structure and adapt to each dataset's format!

---

## Bottom Line

**Data parsing** = Making all datasets speak the same language

- **Input**: 8 different dataset formats (downloaded)
- **Process**: 8 parser classes (you write these)
- **Output**: UnifiedSample format (same for all)
- **Result**: Training code works with ALL datasets!

**After parsers are done**: You can run experiments immediately!

---

*Last updated: 2026-02-07*
