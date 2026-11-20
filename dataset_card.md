# Creative Swearing Generator — Fine-Tuning Dataset

**Repository:** `Abdulhadi446/ml-swearer`  
**Purpose:** Style-conditioned creative insult generation (fine-tuning dataset)  
**License:** Mixed — see `data/creative_insults_metadata.json`

## Dataset overview

| Split | Records |
|-------|---------|
| `train.jsonl` | 448 |
| `validation.jsonl` | 56 |
| `test.jsonl` | 57 |
| **Total (combined.jsonl)** | **561** |

## Styles (contexts)

| Context | Records | Origin |
|---------|---------|--------|
| `grose-1785` | 116 | Grose 1785 (public domain) + synthetic |
| `modern-british` | 110 | British TV/comedy + synthetic |
| `roastme` | 112 | RoastMe style + synthetic |
| `shakespearean` | 97 | Shakespeare/Wikisource + synthetic |
| `absurdist` | 103 | Curated absurdist + synthetic |
| `veep-style` | 11 | Veep TV drama |
| `humor-theory` | 8 | Wikipedia: Insult theory |
| `flyting` | 4 | Scottish/Norse flyting |

## Format (fine-tuning)

Each line in `combined.jsonl` / `train.jsonl` / `validation.jsonl` / `test.jsonl`:

```jsonl
{"context": "shakespearean", "insult": "Thou art a boil, a plague-sore, an embossed carbuncle.", "tags": ["archaic","medical","shakespeare"]}
```

Fine-tuning target: map `context` → `insult`.

### Instruction format

Also provided: `*_instruct.jsonl` (instruction / input / output):

```jsonl
{"instruction": "Generate a creative shakespearean-style insult.", "input": "Context: shakespearean. Tags: archaic, medical", "output": "Thou art a boil, a plague-sore..."}
```

## Training

```bash
pip install transformers torch

python train.py \
  --data data/combined.jsonl \
  --model distilgpt2 \
  --epochs 3 \
  --output-dir ./model
```

`train.py` loads `combined.jsonl`, builds context→insult pairs, and fine-tunes a causal LM.

## Example inference

Prompt: `[shakespearean] archaic, medical`  
Output: `Thou art a boil, a plague-sore, an embossed carbuncle in corrupted flesh.`

Prompt: `[modern-british]`  
Output: `Absolute knob-end.`

## Citation / credit

If you use this dataset, credit `Abdulhadi446/ml-swearer`.
