# Swearing Benchmark (Sewer Benchmark)

A small held-out evaluation set for **style-conditioned creative insult generation**.

Given a style register, a model must produce one insult in that register. The
benchmark is deliberately small and hand-checked: it measures whether a model has
picked up the *register* (archaic diction, British absurdism, roast comedy, Veep
escalation) rather than whether it can memorise a corpus.

- **27 cases** across **7 contexts**
- Every case is held out from `dataset/train.jsonl`
- Each case: `id`, `context`, `reference_insult`, `tags`

## Contexts

| Context | Cases |
|---------|-------|
| `absurdist` | 5 |
| `grose-1785` | 5 |
| `modern-british` | 5 |
| `roastme` | 5 |
| `shakespearean` | 5 |
| `flyting` | 1 |
| `veep-style` | 1 |

## Files

| File | Purpose |
|------|---------|
| `sewer_benchmark.json` | the 27 benchmark cases |
| `eval_sewer.py` | evaluator (transformers / mock backends) |

## Running

```bash
pip install -r ../requirements.txt

# merged model
python eval_sewer.py --model ../model-merged

# LoRA adapter from train_qlora.py
python eval_sewer.py --model ../model --base-model Qwen/Qwen3.5-0.8B-Base

# no model needed — pipeline smoke test
python eval_sewer.py --backend mock
```

`--prompt-format` must match training. `train_qlora.py` defaults to `plain`, so the
evaluator defaults to `plain` too.

## Metrics

| Metric | Meaning |
|--------|---------|
| `exact_match` | normalized string equality with the reference insult |
| `token_f1` | token overlap (precision/recall harmonic mean) |
| `rouge_l_f1` | longest-common-subsequence F1 |

Automatic overlap metrics under-reward creative writing: two good insults in the
same register can share almost no tokens. Use them for regression tracking, and read
the actual generations (`cases` in the output JSON) for quality.

## Output

```json
{
  "summary": {"n_cases": 27, "exact_match": 0.0, "mean_token_f1": 0.0, "mean_rouge_l": 0.0},
  "by_context": {"shakespearean": {"n_cases": 5, "exact_match": 0.0}},
  "cases": [
    {
      "id": "absurdist_you_have_the_moral_c",
      "context": "absurdist",
      "reference": "you have the moral compass of a shopping cart with one broken wheel",
      "prediction": "...",
      "exact_match": 0,
      "token_f1": 0.0,
      "rouge_l_f1": 0.0
    }
  ]
}
```

## Baseline protocol

Report three numbers so the benchmark says something:

1. **Base model** — `Qwen/Qwen3.5-0.8B-Base`, no fine-tuning
2. **Fine-tuned** — same model after `train_qlora.py`
3. **Random context shuffle** — same model, wrong style requested (sanity check that
   the model is actually conditioning on the context)

A useful fine-tune beats its base model and degrades when the context is shuffled.

## Citation

```
@misc{ml-swearer-bench,
  title  = {Swearing Benchmark (Sewer Benchmark)},
  author = {Abdulhadi446},
  year   = {2026},
  url    = {https://github.com/Abdulhadi446/ml-swearer/tree/master/benchmark}
}
```
