# Creative Swearing Generator — Fine-Tuning Dataset

Style-conditioned creative insult generation: given a style register, produce an
inventive insult in that register. Intended as a **fine-tuning dataset**, not as a
profanity dump — the goal is wit, not abuse.

**License:** mixed, see `creative_insults_metadata.json` (public domain for
Grose/Shakespeare-derived material, CC BY-SA 4.0 for Wikipedia-derived material,
original work for synthetic entries).

## Files

| File | Rows | Purpose |
|------|------|---------|
| `combined.jsonl` | 561 | full dataset, deduplicated |
| `train.jsonl` | 448 | 80% split |
| `validation.jsonl` | 56 | 10% split |
| `test.jsonl` | 57 | 10% split |
| `train_instruct.jsonl` | 448 | same rows, instruction format |
| `validation_instruct.jsonl` | 56 | instruction format |
| `test_instruct.jsonl` | 57 | instruction format |
| `creative_insults_metadata.json` | — | per-context counts + licensing |
| `raw/` | — | per-style source files before merging |

## Contexts

| Context | Records | Origin |
|---------|---------|--------|
| `grose-1785` | 116 | Francis Grose, *A Classical Dictionary of the Vulgar Tongue* (1785) + synthetic |
| `roastme` | 112 | r/RoastMe register + synthetic |
| `modern-british` | 110 | British comedy register + synthetic |
| `absurdist` | 103 | curated surreal/absurdist + synthetic |
| `shakespearean` | 97 | Shakespeare / Wikisource + synthetic |
| `veep-style` | 11 | Veep-style escalation and self-correction |
| `humor-theory` | 8 | Wikipedia: Insult — response typology |
| `flyting` | 4 | Scottish/Norse flyting tradition |

## Record format

```jsonl
{"context": "shakespearean", "insult": "Thou art a boil, a plague-sore, an embossed carbuncle in corrupted flesh.", "tags": ["archaic", "medical", "extended-metaphor", "shakespeare"]}
```

- `context` — style register (the conditioning signal)
- `insult` — target text
- `tags` — style hints (may be generic, e.g. `["style", "insult"]`, for synthetic rows)

## Instruction format

```jsonl
{"instruction": "Generate a creative shakespearean-style insult.", "input": "Context: shakespearean. Tags: archaic, medical", "output": "Thou art a boil, a plague-sore..."}
```

## Prompt format used by `train_qlora.py`

The default `plain` format keeps training and evaluation identical:

```
[shakespearean] archaic, medical
Thou art a boil, a plague-sore, an embossed carbuncle in corrupted flesh.
```

Everything after the first newline is the supervised target; the prompt tokens are
masked out of the loss.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("json", data_files="train.jsonl", split="train")
print(ds[0])
```

## Usage

```bash
git clone https://github.com/Abdulhadi446/ml-swearer
cd ml-swearer
pip install -r requirements.txt
python train_qlora.py --model Qwen/Qwen3.5-0.8B-Base --output-dir ./model
```

## Citation

```
@misc{ml-swearer,
  title  = {Creative Swearing Generator: a style-conditioned insult fine-tuning dataset},
  author = {Abdulhadi446},
  year   = {2026},
  url    = {https://github.com/Abdulhadi446/ml-swearer}
}
```
