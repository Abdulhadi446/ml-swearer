# Creative Swearing Generator

A **creative swearing generator** — a model trained to produce *inventive, stylized* insults rather than raw abuse.

## Data

`data/combined.jsonl` — 561 unique records across 8 style contexts (single consolidated file).

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

## Generate / add more styles

Five subagents each produced 100 synthetic entries (shakespearean, grose-1785, modern-british, roastme, absurdist). The generator script `data/raw/fetch_wikisource.py` shows the source-collection pattern.

## Training

Use `context` as the style-conditioned prompt and `insult` as the target. See `data/raw/fetch_wikisource.py` for the collection script.

### Format

```jsonl
{"context": "shakespearean", "insult": "Thou art a boil, a plague-sore, an embossed carbuncle in corrupted flesh.", "tags": ["archaic", "medical", "extended-metaphor", "shakespeare"]}
{"context": "roastme", "insult": "You look like a 'before' photo in a hair transplant ad.", "tags": ["visual", "self-deprecating", "roastme"]}
```

### Blocked sources

- **Project Gutenberg** — down for maintenance (503 every request)
- **Bartleby** — Cloudflare challenge wall
- **websearch** — returning HTTP 402 (payment required)
- **Wikisource** — works, but returns disambiguation pages, not raw play text directly

## To scale

1. Wait for Gutenberg back up → scrape full play text, pull insult-heavy dialogue lines
2. Reddit Pushshift API → `r/rareinsults` top posts + `r/RoastMe` comments (tens of thousands)
3. Kaggle datasets → RoastMe CSV, insult corpora
4. Wikisource resolve redirects → follow redirects to actual `Hamlet_(Shakespeare)` pages

## Training

See `data/raw/fetch_wikisource.py` for the collection script. For training, use the `context` field as the style-conditioned prompt and the `insult` field as the target.