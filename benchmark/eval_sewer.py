#!/usr/bin/env python3
"""
Swearing Benchmark (a.k.a. Sewer Benchmark) evaluator.

Scores a fine-tuned (or base) model against ``sewer_benchmark.json`` — held-out
style-conditioned insult prompts — using exact match, token F1 and Rouge-L F1.

Usage
-----
    # evaluate a merged model
    python benchmark/eval_sewer.py --model ./model-merged

    # evaluate a LoRA adapter directory produced by train_qlora.py
    python benchmark/eval_sewer.py --model ./model --base-model Qwen/Qwen3.5-0.8B-Base

    # smoke test with no model at all
    python benchmark/eval_sewer.py --backend mock

The prompt format must match how the model was trained. ``train_qlora.py``
defaults to ``plain``, which is what this script uses by default.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import string
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = BENCH_DIR / "sewer_benchmark.json"


# ---------------------------------------------------------------------------
# Text normalization and metrics
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)
    return " ".join(text.split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def exact_match(pred: str, ref: str) -> int:
    return int(normalize(pred) == normalize(ref))


def token_f1(pred: str, ref: str) -> float:
    pt, rt = tokens(pred), tokens(ref)
    if not pt or not rt:
        return 0.0
    common = collections.Counter(pt) & collections.Counter(rt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(rt)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(pred: str, ref: str) -> float:
    pt, rt = tokens(pred), tokens(ref)
    if not pt or not rt:
        return 0.0
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
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Prompt formatting (mirrors train_qlora.py)
# ---------------------------------------------------------------------------
def plain_prompt(context: str, tags: list[str]) -> str:
    hint = ", ".join(tags[:2])
    return f"[{context}] {hint}".rstrip()


def chat_prompt(context: str, tags: list[str], tokenizer) -> str:
    hint = ", ".join(tags[:2])
    content = f"Generate a creative {context}-style insult."
    if hint:
        content += f" Style hints: {hint}."
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class Backend:
    def generate(self, context: str, tags: list[str]) -> str:
        raise NotImplementedError


class TransformersBackend(Backend):
    def __init__(self, model_path: str, base_model: str | None = None,
                 prompt_format: str = "plain", max_new_tokens: int = 48,
                 temperature: float = 0.9, top_p: float = 0.95,
                 load_in_4bit: bool = False, seed: int = 42):
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError:
            raise SystemExit("Install transformers and torch: pip install transformers torch")

        self.torch = torch
        self.prompt_format = prompt_format
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(base_model or model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: dict = {"device_map": "auto" if torch.cuda.is_available() else None}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif torch.cuda.is_available():
            kwargs["torch_dtype"] = (
                torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            )

        model = self._load_model(model_path, kwargs)

        # Adapter directory? Wrap it around the base model.
        adapter_file = Path(model_path) / "adapter_config.json"
        if adapter_file.exists():
            try:
                from peft import PeftModel
            except ImportError:
                raise SystemExit("This is a LoRA adapter; install peft: pip install peft")
            model = PeftModel.from_pretrained(model, model_path)

        model.eval()
        self.model = model

    def _load_model(self, model_path: str, kwargs: dict):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        for loader in (AutoModelForImageTextToText, AutoModelForCausalLM):
            try:
                return loader.from_pretrained(model_path, **kwargs)
            except (ImportError, ValueError, KeyError, OSError) as exc:
                last = exc
        raise SystemExit(f"Could not load {model_path}: {last}")

    def generate(self, context: str, tags: list[str]) -> str:
        if self.prompt_format == "chat":
            prompt = chat_prompt(context, tags, self.tokenizer)
        else:
            prompt = plain_prompt(context, tags) + "\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                top_p=self.top_p,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        gen = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return gen.strip().split("\n")[0].strip()


class MockBackend(Backend):
    """Deterministic dummy for smoke-testing without a model."""

    def generate(self, context: str, tags: list[str]) -> str:
        return f"[mock] you're a {context}-style placeholder"


def load_backend(args) -> Backend:
    if args.backend == "transformers":
        return TransformersBackend(
            model_path=args.model,
            base_model=args.base_model,
            prompt_format=args.prompt_format,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            load_in_4bit=args.load_in_4bit,
        )
    return MockBackend()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(backend: Backend, benchmark: list[dict], limit: int | None = None) -> list[dict]:
    cases = benchmark if limit is None else benchmark[:limit]
    results = []
    for i, case in enumerate(cases, 1):
        context = case["context"]
        tags = case.get("tags", [])
        reference = case["reference_insult"]
        prediction = backend.generate(context, tags)
        results.append({
            "id": case.get("id", f"{context}_{i}"),
            "context": context,
            "reference": reference,
            "prediction": prediction,
            "exact_match": exact_match(prediction, reference),
            "token_f1": round(token_f1(prediction, reference), 4),
            "rouge_l_f1": round(rouge_l_f1(prediction, reference), 4),
        })
    return results


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {"n_cases": 0, "exact_match": 0.0, "mean_token_f1": 0.0, "mean_rouge_l": 0.0}
    return {
        "n_cases": n,
        "exact_match": round(sum(r["exact_match"] for r in results) / n, 4),
        "mean_token_f1": round(sum(r["token_f1"] for r in results) / n, 4),
        "mean_rouge_l": round(sum(r["rouge_l_f1"] for r in results) / n, 4),
    }


def per_context(results: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for r in results:
        grouped.setdefault(r["context"], []).append(r)
    return {ctx: aggregate(rs) for ctx, rs in sorted(grouped.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a model on the Swearing Benchmark")
    ap.add_argument("--model", default="./model", help="fine-tuned model or LoRA adapter dir")
    ap.add_argument("--base-model", default=None,
                    help="base model id when --model is a LoRA adapter")
    ap.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    ap.add_argument("--backend", default="transformers", choices=["transformers", "mock"])
    ap.add_argument("--prompt-format", default="plain", choices=["plain", "chat"],
                    help="must match the format used in training")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N cases")
    ap.add_argument("--output", default="sewer_results.json")
    args = ap.parse_args()

    bench_path = Path(args.benchmark)
    if not bench_path.exists():
        raise SystemExit(f"Benchmark file not found: {bench_path}")

    benchmark = json.loads(bench_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(benchmark)} benchmark cases from {bench_path}")

    backend = load_backend(args)
    print(f"Backend: {args.backend} ({args.model})")

    results = evaluate(backend, benchmark, limit=args.limit)
    summary = aggregate(results)
    by_ctx = per_context(results)

    print("\n=== Swearing Benchmark Results ===")
    print(f"Cases:         {summary['n_cases']}")
    print(f"Exact match:   {summary['exact_match']}")
    print(f"Mean Token F1: {summary['mean_token_f1']}")
    print(f"Mean Rouge-L:  {summary['mean_rouge_l']}")
    print("\nBy context:")
    for ctx, m in by_ctx.items():
        print(f"  {ctx:20s} n={m['n_cases']:2d} exact={m['exact_match']:.2f} "
              f"f1={m['mean_token_f1']:.2f} rougeL={m['mean_rouge_l']:.2f}")

    out = {"summary": summary, "by_context": by_ctx, "cases": results}
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
