#!/usr/bin/env python3
"""
Train a creative swearing generator conditioned on context/style.

Usage:
    python train.py --data dataset/combined.jsonl --epochs 3 --output-dir ./model
"""
import argparse
import json
import os


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed line in {path}: {exc}")
    return records


def build_examples(records):
    """Build prompt→target pairs: context + optional tag hint -> insult."""
    examples = []
    for r in records:
        ctx = r.get("context", "").strip()
        ins = r.get("insult", "").strip()
        tags = r.get("tags", [])
        tag_hint = ", ".join(tags[:2]) if tags else ""
        prompt = f"[{ctx}] {tag_hint}".strip()
        examples.append({"prompt": prompt, "target": ins})
    return examples


def build_dataset(examples, tokenizer, max_len=64):
    texts = [ex["prompt"] + "\n" + ex["target"] for ex in examples]
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=max_len, return_tensors="pt")
    labels = enc["input_ids"].clone()
    # Mask prompt tokens so loss only on target
    for i, ex in enumerate(examples):
        plen = len(tokenizer(ex["prompt"])["input_ids"])
        labels[i, :plen] = -100
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/combined.jsonl")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--output-dir", default="./model")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    except ImportError:
        raise SystemExit("Install transformers: pip install transformers torch")

    records = load_jsonl(args.data)
    examples = build_examples(records)
    print(f"Loaded {len(examples)} examples from {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    class Dataset:
        def __init__(self, enc):
            self.enc = enc
        def __len__(self):
            return self.enc["input_ids"].shape[0]
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}

    enc = build_dataset(examples, tokenizer, args.max_len)
    dataset = Dataset(enc)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=dataset)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()