# src/data/CLAUDE.md — Data Pipeline Instructions

## Unified Data Format

All 5 SemEval tasks are converted to a common JSON Lines format. This enables multi-task training with a single DataLoader.

```python
@dataclass
class UnifiedSample:
    task: str                    # "semeval2014t7", "semeval2015t14", etc.
    task_type: str               # "ner", "temporal_re", "qa_ranking"
    task_level: int              # 1 = low-level (NER), 2 = high-level (RE/QA)
    input_text: str              # raw clinical text
    labels: dict                 # task-specific labels
    metadata: dict               # doc_id, sent_id, source
    token_count: int             # number of tokens (for RQ5 tracking)
```

## Task-Specific Parsers

### semeval2014t7.py
- Parse BRAT annotation format (.ann files)
- Entity types: DISORDER (with CUI normalization)
- Output: BIO-tagged sequences + CUI mapping
- Standard BIO tagging is fine here

### semeval2015t14.py — SPECIAL HANDLING REQUIRED
- **Discontiguous entities**: "left and right atrial enlargement"
- DO NOT use standard BIO tagging
- Use span-based format: list of (start, end, type) tuples with linking IDs
- Requires SpanClassificationHead in models/heads.py

```python
# Example discontiguous entity:
{
    "entity_id": "E1",
    "type": "DISORDER",
    "spans": [(0, 4), (9, 31)],  # "left" + "atrial enlargement"
    "text": "left ... atrial enlargement"
}
```

### semeval2016t12.py
- Parse temporal relations (TLINK, EVENT, TIMEX)
- Subtasks: EVENT detection, TIMEX detection, TLINK classification
- Output: (head_entity, tail_entity, relation_type) triples

### semeval2017t3.py
- QA pair ranking format
- Input: question + candidate answer
- Output: relevance score (for MAP/MRR evaluation)
- This is retrieval/ranking, NOT generative QA

### semeval2021t6.py
- Combined NER + Relation Extraction
- Has BOTH Level 1 (NER) and Level 2 (RE) components
- Parse both and assign appropriate task_level

## MultiTaskDataLoader

### Sampling Strategies

```python
class MultiTaskBatchSampler:
    """
    Three sampling strategies for multi-task training:
    
    1. Proportional: P(task) ∝ |D_task|
       Larger datasets sampled more. Risk: small tasks starved.
       
    2. Uniform: P(task) = 1/T
       Equal probability. Risk: large tasks undersampled.
       
    3. Temperature: P(task) ∝ |D_task|^(1/τ)
       τ=1 = proportional, τ=∞ = uniform. τ=2 is usually best.
    """
```

### Token Counting (CRITICAL for RQ5)

```python
class TokenTracker:
    """Track cumulative tokens per task during training."""
    
    def __init__(self):
        self.tokens = defaultdict(int)
        
    def update(self, task_name: str, batch: dict):
        n_tokens = (batch['attention_mask'] != 0).sum().item()
        self.tokens[task_name] += n_tokens
        
    def total(self) -> int:
        return sum(self.tokens.values())
        
    def report(self) -> dict:
        return dict(self.tokens)
```

## Data Access

- **PhysioNet tasks (2014-T7, 2015-T14, 2016-T12)**: Require CITI training + DUA. Apply at physionet.org IMMEDIATELY.
- **Public tasks (2017-T3, 2021-T6)**: Directly downloadable.
- **Start with public tasks** while waiting for PhysioNet approval.

## Sequence Length Optimization

Run `utils/analyze_seq_lengths.py` on each task BEFORE training:
- If 95th percentile < 256 tokens → use max_seq_len=256 (saves 40% VRAM)
- If 95th percentile < 384 tokens → use max_seq_len=384 (saves 25% VRAM)
- Only use 512 if data actually needs it
