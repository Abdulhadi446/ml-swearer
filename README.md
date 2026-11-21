# Creative Swearing Generator

A **creative swearing generator** — a model trained to produce *inventive, stylized* insults rather than raw abuse.

## Data

`data/combined.jsonl` — 561 unique records across 8 style contexts (single consolidated file).

### Splits (fine-tuning)

- `data/train.jsonl` — 448 records
- `data/validation.jsonl` — 56 records
- `data/test.jsonl` — 57 records

Instruct format also provided in `*_instruct.jsonl` (`instruction` / `input` / `output`).

See `dataset_card.md` for the full fine-tuning dataset card.

### Source context breakdown

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
| **Total** | **561** | |

### Format

```jsonl
{"context": "shakespearean", "insult": "Thou art a boil, a plague-sore, an embossed carbuncle in corrupted flesh.", "tags": ["archaic", "medical", "extended-metaphor", "shakespeare"]}
{"context": "roastme", "insult": "You look like a 'before' photo in a hair transplant ad.", "tags": ["visual", "self-deprecating", "roastme"]}
```

Each record has three fields:
- `context` — style register of the insult
- `insult` — the insult text (target)
- `tags` — style hints

## Training

Use `context` as the style-conditioned prompt and `insult` as the target. Train with the script below:

```bash
python train.py --data data/combined.jsonl --epochs 3 --output-dir ./model
```

See `train.py` for full CLI options. The model learns to map `context` → `insult`, so at inference time you pass a style (e.g. `shakespearean`) and it generates a creative insult in that register.

## Generate / add more styles

Five subagents each produced 100 synthetic entries (shakespearean, grose-1785, modern-british, roastme, absurdist). The generator script `data/raw/fetch_wikisource.py` shows the source-collection pattern.

## Blocked sources

- **Project Gutenberg** — down for maintenance (503 every request)
- **Bartleby** — Cloudflare challenge wall
- **websearch** — returning HTTP 402 (payment required)
- **Wikisource** — works, but returns disambiguation pages, not raw play text directly

## To scale

1. Wait for Gutenberg back up → scrape full play text, pull insult-heavy dialogue lines
2. Reddit Pushshift API → `r/rareinsults` top posts + `r/RoastMe` comments (tens of thousands)
3. Kaggle datasets → RoastMe CSV, insult corpora
4. Wikisource resolve redirects → follow redirects to actual `Hamlet_(Shakespeare)` pages

## Benchmark: Sewer

`data/sewer_benchmark.json` — 27 held-out test cases across 7 contexts for evaluating how well a fine-tuned model generates creative insults.

### Run evaluation

```bash
# With transformers backend (needs a fine-tuned model)
python eval_sewer.py \
  --model ./model \
  --benchmark data/sewer_benchmark.json \
  --output sewer_results.json

# Smoke test with mock backend
python eval_sewer.py \
  --backend mock \
  --model ./model \
  --output sewer_results.json
```

Metrics reported per run:
- **Exact match** — normalized equality between generated and reference insult
- **Token F1** — token-level overlap
- **Rouge-L F1** — longest common subsequence overlap
- **By-context breakdown** — performance per style register

### Example output

```
Cases: 27
Exact Match: 0.148
Mean Token F1: 0.412
Mean Rouge-L: 0.387

By Context:
  absurdist              n=  5 exact=0.20 f1=0.45 rougeL=0.41
  flyting                n=  1 exact=0.00 f1=0.00 rougeL=0.00
  grose-1785             n=  5 exact=0.00 f1=0.33 rougeL=0.31
  modern-british         n=  5 exact=0.20 f1=0.60 rougeL=0.58
  roastme                n=  5 exact=0.40 f1=0.72 rougeL=0.68
  shakespearean          n=  5 exact=0.00 f1=0.38 rougeL=0.35
  veep-style             n=  1 exact=0.00 f1=0.25 rougeL=0.22
```

## License

Mixed — see `data/creative_insults_metadata.json` per context. Public domain for Grose/Shakespeare-derived content; synthetic entries are original work; Wikipedia-derived content is CC BY-SA 4.0.
