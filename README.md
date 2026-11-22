# Creative Swearing Generator

A **style-conditioned creative insult generator**: fine-tune a small language model to
produce *inventive, stylized* insults in a requested register instead of generic abuse.

Two self-contained artifacts live here, each uploadable on its own:

| Folder | What it is | Upload as |
|--------|------------|-----------|
| `dataset/` | 561-record style-conditioned insult dataset + splits | HF **dataset** repo |
| `benchmark/` | **Swearing Benchmark** — 27 held-out cases + evaluator | HF **dataset** repo |
| `model-merged/` (generated) | the fine-tuned model | HF **model** repo |

## Layout

```
.
├── dataset/
│   ├── README.md                     dataset card
│   ├── combined.jsonl                561 records (deduplicated)
│   ├── train.jsonl                   448
│   ├── validation.jsonl              56
│   ├── test.jsonl                    57
│   ├── *_instruct.jsonl              instruction format variants
│   ├── creative_insults_metadata.json
│   └── raw/                          per-style sources before merging
├── benchmark/
│   ├── README.md                     benchmark card
│   ├── sewer_benchmark.json          27 held-out cases
│   └── eval_sewer.py                 evaluator
├── train_qlora.py                    QLoRA fine-tune of Qwen3.5-0.8B-Base
├── train.py                          small baseline (distilgpt2)
├── requirements.txt
└── README.md
```

## Dataset

561 unique records across 8 registers, split 80/10/10.

| Context | Records | Origin |
|---------|---------|--------|
| `grose-1785` | 116 | Grose 1785 (public domain) + synthetic |
| `roastme` | 112 | RoastMe register + synthetic |
| `modern-british` | 110 | British comedy register + synthetic |
| `absurdist` | 103 | curated absurdist + synthetic |
| `shakespearean` | 97 | Shakespeare / Wikisource + synthetic |
| `veep-style` | 11 | Veep-style escalation |
| `humor-theory` | 8 | Wikipedia: Insult |
| `flyting` | 4 | Scottish/Norse flyting |

```jsonl
{"context": "shakespearean", "insult": "Thou art a boil, a plague-sore, an embossed carbuncle in corrupted flesh.", "tags": ["archaic", "medical", "shakespeare"]}
```

Full details: [`dataset/README.md`](dataset/README.md).

## Swearing Benchmark

27 held-out cases across 7 contexts (`absurdist`, `flyting`, `grose-1785`,
`modern-british`, `roastme`, `shakespearean`, `veep-style`), scored with exact match,
token F1 and Rouge-L, with a per-context breakdown.

Full details: [`benchmark/README.md`](benchmark/README.md).

## Fine-tuning (QLoRA)

`train_qlora.py` fine-tunes **`Qwen/Qwen3.5-0.8B-Base`** with 4-bit NF4 QLoRA:
LoRA `r=16`, `alpha=32`, all language-model projections adapted, prompt tokens masked
out of the loss. A 16 GB GPU is enough.

> Qwen3.5-0.8B is a vision-language checkpoint, so the text weights live under
> `model.language_model.*`. The script loads the multimodal class and adapts only the
> language model, which is what makes the weights load correctly.

### On Kaggle

```bash
!pip install -q -U "transformers>=5.20" peft accelerate bitsandbytes

%cd /kaggle/working
!git clone https://github.com/Abdulhadi446/ml-swearer
%cd ml-swearer

!python train_qlora.py \
    --output-dir /kaggle/working/qwen35-swearer-lora \
    --epochs 3 --batch-size 8 --grad-accum 2 --lr 2e-4
```

That writes a LoRA adapter. Add `--merge-and-save` to also fold the adapter into a
fresh fp16 copy of the base, giving one self-contained model folder to upload:

```bash
!python train_qlora.py \
    --output-dir /kaggle/working/qwen35-swearer-lora \
    --merge-and-save --merge-dir /kaggle/working/qwen35-swearer \
    --epochs 3 --batch-size 8 --grad-accum 2
```

Push straight to the Hub:

```bash
!python train_qlora.py --merge-and-save --push-to-hub <user>/qwen35-swearer
```

### Local checks (no GPU needed)

```bash
python train_qlora.py --check-data    # validate the dataset files
python train_qlora.py --dry-run       # tokenise + mask, stop before loading the model
```

### Prompt format

Training and evaluation share one format, so scores mean something:

```
[shakespearean] archaic, medical
Thou art a boil, a plague-sore, an embossed carbuncle in corrupted flesh.
```

Everything after the first newline is the target; the prompt is masked. `--prompt-format chat`
switches to the tokenizer's chat template if you fine-tune the instruct checkpoint
(`Qwen/Qwen3.5-0.8B`) instead of the base model.

### Useful flags

| Flag | Default | Notes |
|------|---------|-------|
| `--model` | `Qwen/Qwen3.5-0.8B-Base` | any compatible HF id |
| `--epochs` | `3` | 448 rows, so epochs are cheap |
| `--batch-size` | `8` | lower to 4 if you hit OOM |
| `--grad-accum` | `2` | effective batch 16 |
| `--lr` | `2e-4` | LoRA-typical |
| `--max-seq-len` | `256` | insults are short |
| `--lora-r` / `--lora-alpha` | `16` / `32` | |
| `--no-4bit` | off | plain fp16 LoRA |
| `--generate-samples N` | `0` | print generations after training |
| `--limit N` | `0` | smoke-test on N rows |

On GPUs below compute capability 7.0 (e.g. Kaggle's P100) bitsandbytes 4-bit is
unavailable; the script detects this and falls back to plain fp16 LoRA.

## Evaluation

```bash
# merged model
python benchmark/eval_sewer.py --model ./model-merged

# LoRA adapter
python benchmark/eval_sewer.py --model ./model --base-model Qwen/Qwen3.5-0.8B-Base

# pipeline smoke test, no model
python benchmark/eval_sewer.py --backend mock
```

Reports overall and per-context exact match / token F1 / Rouge-L and writes
`sewer_results.json` with every generation.

## Baseline

`train.py` is a small dependency-light baseline (default `distilgpt2`) that trains on
the same `context → insult` mapping:

```bash
python train.py --data dataset/combined.jsonl --epochs 3 --output-dir ./baseline
```

## License

Mixed — see `dataset/creative_insults_metadata.json`. Public domain for
Grose/Shakespeare-derived content, CC BY-SA 4.0 for Wikipedia-derived content,
original work for synthetic entries. Fine-tuned weights inherit the
`Qwen/Qwen3.5-0.8B-Base` Apache-2.0 license.

## Warning

The dataset and models here generate deliberately rude, insulting text. It is built
for comedy and creative-writing research. Do not deploy it to harass people.
