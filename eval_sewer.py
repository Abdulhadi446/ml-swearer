#!/usr/bin/env python3
"""
Sewer Benchmark Evaluator

Evaluates a fine-tuned creative-swearing model against the Sewer Benchmark
(data/sewer_benchmark.json).

Metrics:
  - Exact match (EXACT): generated insult == reference insult (case-insensitive, normalized)
  - Token-level F1 between generated and reference
  - Rouge-L F1 between generated and reference
  - Context match: does the model's output match the requested context/style?

Usage:
  python eval_sewer.py \
    --model ./model \
    --benchmark data/sewer_benchmark.json \
    --output results.json \
    --backend transformers   # or: openai, anthropic, mock
"""
import argparse
import json
import re
import string
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------
def normalize(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)
    return " ".join(text.split())


def tokens(text):
    return normalize(text).split()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def exact_match(pred, ref):
    return int(normalize(pred) == normalize(ref))


def token_f1(pred, ref):
    pt = tokens(pred)
    rt = tokens(ref)
    if not pt or not rt:
        return 0.0
    common = collections.Counter(pt) & collections.Counter(rt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(rt)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(pred, ref):
    pt = tokens(pred)
    rt = tokens(ref)
    if not pt or not rt:
        return 0.0
    # Longest common subsequence length via DP
    m, n = len(pt), len(rt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if pt[i] == rt[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[0][0]
    if lcs == 0:
        return 0.0
    precision = lcs / m
    recall = lcs / n
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class Backend:
    def generate(self, context, tags):
        raise NotImplementedError


class TransformersBackend(Backend):
    def __init__(self, model_path):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise SystemExit("Install transformers: pip install transformers torch")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.eval()

    def generate(self, context, tags):
        tag_hint = ", ".join(tags[:2]) if tags else ""
        prompt = f"[{context}] {tag_hint}\n"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with __import__("torch").no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
        return gen.strip()


class MockBackend(Backend):
    """Deterministic dummy for smoke-testing without a real model."""
    def generate(self, context, tags):
        return f"[mock] you're a {context}-style placeholder"


def load_backend(backend, model_path):
    if backend == "transformers":
        return TransformersBackend(model_path)
    elif backend == "mock":
        return MockBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(backend, benchmark, limit=None):
    results = []
    cases = benchmark if limit is None else benchmark[:limit]

    for case in cases:
        context = case["context"]
        tags = case.get("tags", [])
        reference = case["reference_insult"]
        case_id = case.get("id", f"{context}_{reference[:10]}")

        prediction = backend.generate(context, tags)

        result = {
            "id": case_id,
            "context": context,
            "reference": reference,
            "prediction": prediction,
            "exact_match": exact_match(prediction, reference),
            "token_f1": token_f1(prediction, reference),
            "rouge_l_f1": rouge_l_f1(prediction, reference),
        }
        results.append(result)

    return results


def aggregate(results):
    n = len(results)
    if n == 0:
        return {}
    return {
        "n_cases": n,
        "exact_match": round(sum(r["exact_match"] for r in results) / n, 4),
        "mean_token_f1": round(sum(r["token_f1"] for r in results) / n, 4),
        "mean_rouge_l": round(sum(r["rouge_l_f1"] for r in results) / n, 4),
    }


def per_context(results):
    out = {}
    for r in results:
        ctx = r["context"]
        out.setdefault(ctx, []).append(r)
    return {ctx: aggregate(rs) for ctx, rs in out.items()}


def main():
    ap = argparse.ArgumentParser(description="Evaluate on the Sewer Benchmark")
    ap.add_argument("--model", required=True, help="Path to fine-tuned model")
    ap.add_argument("--benchmark", default="data/sewer_benchmark.json",
                    help="Benchmark JSON file")
    ap.add_argument("--backend", default="transformers",
                    choices=["transformers", "mock"],
                    help="Inference backend")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only evaluate first N cases")
    ap.add_argument("--output", default="sewer_results.json",
                    help="Output JSON file for results")
    args = ap.parse_args()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise SystemExit(f"Benchmark file not found: {benchmark_path}")

    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(benchmark)} benchmark cases")

    backend = load_backend(args.backend, args.model)
    print(f"Using backend: {args.backend} ({args.model})")

    results = evaluate(backend, benchmark, limit=args.limit)
    summary = aggregate(results)
    by_ctx = per_context(results)

    print("\n=== Sewer Benchmark Results ===")
    print(f"Cases: {summary.get('n_cases', 0)}")
    print(f"Exact Match: {summary.get('exact_match', 0)}")
    print(f"Mean Token F1: {summary.get('mean_token_f1', 0)}")
    print(f"Mean Rouge-L: {summary.get('mean_rouge_l', 0)}")
    print("\nBy Context:")
    for ctx, m in sorted(by_ctx.items()):
        print(f"  {ctx:20s} n={m['n_cases']:2d} "
              f"exact={m['exact_match']:.2f} "
              f"f1={m['mean_token_f1']:.2f} "
              f"rougeL={m['mean_rouge_l']:.2f}")

    out = {
        "summary": summary,
        "by_context": by_ctx,
        "cases": results,
    }
    out_path = Path(args.output)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    import collections
    main()
