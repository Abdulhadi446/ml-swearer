#!/usr/bin/env python3
"""
QLoRA fine-tuning for Qwen3.5-0.8B-Base on the Swearer dataset.

Trains a style-conditioned creative-insult generator: the model is given a
style context (``[shakespearean] archaic, medical``) and learns to continue
with an insult in that register.

Designed to be cloned and run on Kaggle (single 16 GB GPU is plenty).

Kaggle quick start
------------------
    !pip install -q -U "transformers>=5.20" peft accelerate bitsandbytes
    %cd /kaggle/working
    !git clone https://github.com/Abdulhadi446/ml-swearer
    %cd ml-swearer
    !python train_qlora.py \
        --output-dir /kaggle/working/qwen35-swearer-lora \
        --epochs 3 --batch-size 8 --grad-accum 2

    # merged fp16 model, ready to upload as a model card
    !python train_qlora.py --merge-and-save --merge-dir /kaggle/working/qwen35-swearer

Local (no GPU) sanity check of the data pipeline only:
    python train_qlora.py --check-data

Notes
-----
* Qwen3.5-0.8B is a *vision-language* checkpoint (``Qwen3_5ForConditionalGeneration``)
  whose text weights live under ``model.language_model.*``. We therefore load the
  multimodal class and attach LoRA only to the language-model projections, so the
  text weights actually load. The vision tower is frozen and never used.
* 4-bit (NF4) QLoRA is the default. If the GPU is too old for bitsandbytes
  (< compute capability 7.0, e.g. Kaggle's P100), the script automatically falls
  back to plain fp16 LoRA so the run still succeeds.
* ``--prompt-format plain`` matches ``benchmark/eval_sewer.py`` exactly, so the
  trained model can be scored on the Swearing Benchmark without changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN = REPO_ROOT / "dataset" / "train.jsonl"
DEFAULT_VAL = REPO_ROOT / "dataset" / "validation.jsonl"
DEFAULT_BENCH = REPO_ROOT / "benchmark" / "sewer_benchmark.json"

# Every nn.Linear inside the Qwen3.5 language model.
#   full attention : q_proj k_proj v_proj o_proj
#   GatedDeltaNet  : in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj
#   MLP            : gate_proj up_proj down_proj
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
]

# ---------------------------------------------------------------------------
# Data (stdlib only, so --check-data runs anywhere)
# ---------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  ! skipping malformed line {lineno} in {path.name}: {exc}")
                continue
            if rec.get("insult") and rec.get("context"):
                records.append(rec)
    return records


def tag_hint_of(rec: dict, limit: int = 2) -> str:
    tags = rec.get("tags") or []
    return ", ".join(str(t) for t in tags[:limit])


def plain_prompt(context: str, tag_hint: str) -> str:
    """Must stay identical to benchmark/eval_sewer.py's prompt."""
    return f"[{context}] {tag_hint}".rstrip()


def chat_prompt(context: str, tag_hint: str, tokenizer) -> str:
    content = f"Generate a creative {context}-style insult."
    if tag_hint:
        content += f" Style hints: {tag_hint}."
    messages = [{"role": "user", "content": content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def build_prompt(rec: dict, fmt: str, tokenizer=None) -> str:
    hint = tag_hint_of(rec)
    if fmt == "chat":
        return chat_prompt(rec["context"], hint, tokenizer)
    return plain_prompt(rec["context"], hint)


def check_data(train_path: Path, val_path: Path) -> None:
    """Dependency-free validation of the dataset files."""
    train = read_jsonl(train_path)
    val = read_jsonl(val_path) if val_path.exists() else []
    print(f"train : {len(train):5d} records  ({train_path})")
    print(f"valid : {len(val):5d} records  ({val_path})")

    contexts: dict[str, int] = {}
    for rec in train:
        contexts[rec["context"]] = contexts.get(rec["context"], 0) + 1
    print("\ncontexts in train split:")
    for ctx, n in sorted(contexts.items(), key=lambda kv: -kv[1]):
        print(f"  {ctx:20s} {n:4d}")

    if train:
        sample = train[0]
        print("\nexample prompt :", repr(plain_prompt(sample["context"], tag_hint_of(sample))))
        print("example target :", repr(sample["insult"][:90]))

    # Duplicate check across the whole split.
    seen = {}
    dupes = 0
    for rec in train:
        key = (rec["context"], rec["insult"].strip().lower())
        if key in seen:
            dupes += 1
        seen[key] = True
    print(f"\nduplicate (context, insult) pairs in train: {dupes}")


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
def encode_example(tokenizer, prompt: str, completion: str, max_len: int) -> dict:
    """Concatenate prompt+completion, masking the prompt out of the loss."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    completion_ids.append(tokenizer.eos_token_id)

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    return {"input_ids": input_ids, "labels": labels}


class TokenizedDataset:
    """Minimal torch-free dataset wrapper (avoids the `datasets` dependency)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def make_collator(pad_token_id: int):
    import torch

    def collate(batch: list[dict]) -> dict:
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids, labels, attention = [], [], []
        for row in batch:
            pad = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_token_id] * pad)
            labels.append(row["labels"] + [-100] * pad)
            attention.append([1] * len(row["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    return collate


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def pick_dtype(torch):
    """Select dtype based on hardware availability."""
    # TPU supports bfloat16 natively
    if getattr(torch, "xla_available", False):
        return torch.bfloat16
    # GPU support
    if getattr(torch, "cuda", False) and hasattr(torch, "is_bf16_supported"):
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float16


def gpu_supports_4bit(torch) -> bool:
    if not getattr(torch, "cuda", False):
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 7
    except Exception:
        return False


def load_model(model_id: str, load_in_4bit: bool, dtype, device_map: str):
    import torch
    from transformers import AutoModelForImageTextToText

    kwargs: dict = {"device_map": device_map, "torch_dtype": dtype}

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    try:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    except (ImportError, ValueError, KeyError) as exc:
        # Very old/new transformers may not expose the alias; fall back to the class.
        print(f"  ! AutoModelForImageTextToText failed ({exc}); trying explicit class")
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError:
            raise SystemExit(
                "This transformers build has no Qwen3.5 support. Install a newer one:\n"
                "  pip install -U \"transformers>=5.20\""
            )
        model = Qwen3_5ForConditionalGeneration.from_pretrained(model_id, **kwargs)

    # Text-only training: the vision tower is dead weight. Disable it when the
    # class supports it so it never runs or receives gradients.
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is not None:
        for param in visual.parameters():
            param.requires_grad = False

    return model


def attach_lora(model, args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )

    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
        exclude_modules=r".*visual.*",
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen3.5-0.8B-Base on the Swearer dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base",
                    help="HF model id or local path")
    ap.add_argument("--train-file", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val-file", default=str(DEFAULT_VAL))
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "model"))
    ap.add_argument("--prompt-format", choices=["plain", "chat"], default="plain",
                    help="plain matches benchmark/eval_sewer.py; chat uses the tokenizer template")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True,
                    help="QLoRA 4-bit NF4 (default)")
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                    help="plain LoRA in fp16/bf16")
    ap.add_argument("--gradient-checkpointing", action="store_true", default=True)
    ap.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                    action="store_false")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N training rows (smoke test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--merge-and-save", action="store_true",
                    help="also write a merged fp16 model (adapter folded into base)")
    ap.add_argument("--merge-dir", default=str(REPO_ROOT / "model-merged"))
    ap.add_argument("--generate-samples", type=int, default=0,
                    help="print N sample generations after training")
    ap.add_argument("--benchmark", default=str(DEFAULT_BENCH),
                    help="benchmark JSON used for sample contexts")
    ap.add_argument("--push-to-hub", default="", help="HF repo id, e.g. user/qwen35-swearer")
    ap.add_argument("--private", action="store_true", help="create a private HF repo")
    ap.add_argument("--check-data", action="store_true",
                    help="validate the dataset and exit (no torch needed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="tokenise everything and exit before loading the model")
    args = ap.parse_args()

    train_path = Path(args.train_file).expanduser()
    val_path = Path(args.val_file).expanduser()

    if args.check_data:
        check_data(train_path, val_path)
        return

    if not train_path.exists():
        raise SystemExit(f"Training file not found: {train_path}")

    # ------------------------------------------------------------------ deps
    try:
        from transformers import AutoTokenizer, Trainer, TrainingArguments
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency ({exc.name}). Install with:\n"
            "  pip install -U \"transformers>=5.20\" peft accelerate bitsandbytes"
        )

    # ------------------------------------------------------------------ runtime
    try:
        import torch
    except ImportError:
        if not args.dry_run:
            raise SystemExit(
                "PyTorch is not installed. Install with:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )
        torch = None  # Will be set later if TPU is available

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------ data
    train_records = read_jsonl(train_path)
    val_records = read_jsonl(val_path) if val_path.exists() else []
    if args.limit:
        train_records = train_records[: args.limit]
        val_records = val_records[: max(1, args.limit // 8)]
    if not train_records:
        raise SystemExit(f"No usable records in {train_path}")
    print(f"Loaded {len(train_records)} train / {len(val_records)} validation records")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def build_rows(records: list[dict]) -> list[dict]:
        rows = []
        for rec in records:
            prompt = build_prompt(rec, args.prompt_format, tokenizer)
            rows.append(encode_example(tokenizer, prompt, rec["insult"].strip(),
                                       args.max_seq_len))
        return rows

    train_rows = build_rows(train_records)
    val_rows = build_rows(val_records)
    lengths = [len(r["input_ids"]) for r in train_rows]
    print(f"Tokenised: mean {sum(lengths)/len(lengths):.1f} tokens, "
          f"max {max(lengths)}, truncated {sum(1 for n in lengths if n >= args.max_seq_len)}")

    if args.dry_run:
        print("\n--- dry run sample ---")
        row = train_rows[0]
        kept = [t for t, l in zip(row["input_ids"], row["labels"]) if l != -100]
        print("decoded (loss tokens only):", repr(tokenizer.decode(kept)))
        print("dry run complete; model not loaded")
        return

    # ------------------------------------------------------------------ runtime
    torch.manual_seed(args.seed)
    if not getattr(torch, "cuda", False) and not getattr(torch, "xla_available", False):
        print("! No CUDA or TPU device found — training will be extremely slow on CPU.")
        print("  On Kaggle, enable the GPU accelerator in the notebook settings.")
        print("  For TPU, ensure torch_xla is installed and enabled.")

    dtype = pick_dtype(torch)
    if args.load_in_4bit and not gpu_supports_4bit(torch) and not getattr(torch, "xla_available", False):
        print("! TPU or GPU does not support bitsandbytes 4-bit.")
        print("  Falling back to plain LoRA in", dtype)
        args.load_in_4bit = False

    # ----------------------------------------------------------------- model
    print(f"Loading {args.model} (4bit={args.load_in_4bit}, dtype={dtype}) ...")
    if getattr(torch, "xla_available", False):
        # TPU: no device_map needed, let XLA handle placement
        model = load_model(args.model, args.load_in_4bit, dtype, device_map=None)
    else:
        model = load_model(args.model, args.load_in_4bit, dtype, device_map="auto")

    model = attach_lora(model, args)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    # ------------------------------------------------------------- arguments
    use_bf16 = dtype == torch.bfloat16
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if val_rows else "no",
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="paged_adamw_8bit" if args.load_in_4bit else "adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=TokenizedDataset(train_rows),
        eval_dataset=TokenizedDataset(val_rows) if val_rows else None,
        data_collator=make_collator(tokenizer.pad_token_id),
    )

    print("Starting training ...")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")

    # ---------------------------------------------------------------- samples
    if args.generate_samples:
        generate_samples(model, tokenizer, args)

    # ----------------------------------------------------------------- merge
    if args.merge_and_save:
        merge_and_save(args.model, args.output_dir, args.merge_dir, dtype)

    # ------------------------------------------------------------------ push
    if args.push_to_hub:
        push_to_hub(args.push_to_hub, args.merge_dir if args.merge_and_save else args.output_dir,
                    args.private)


# ---------------------------------------------------------------------------
# Post-training helpers
# ---------------------------------------------------------------------------
def generate_samples(model, tokenizer, args, n: int | None = None) -> None:
    import torch

    n = n or args.generate_samples
    bench_path = Path(args.benchmark)
    if bench_path.exists():
        cases = json.loads(bench_path.read_text(encoding="utf-8"))
        contexts = []
        for case in cases:
            entry = (case["context"], ", ".join(case.get("tags", [])[:2]))
            if entry not in contexts:
                contexts.append(entry)
    else:
        contexts = [("shakespearean", "archaic, medical"), ("roastme", "visual")]

    model.eval()
    device = next(model.parameters()).device
    print("\n=== sample generations ===")
    for ctx, hint in contexts[:n]:
        if args.prompt_format == "chat":
            prompt = chat_prompt(ctx, hint, tokenizer)
        else:
            prompt = plain_prompt(ctx, hint) + "\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
        print(f"  [{ctx}] {hint}\n    -> {text}")

def merge_and_save(base_id: str, adapter_dir: str, out_dir: str, dtype) -> None:
    """Fold the adapter into a fresh fp16/bf16 copy of the base model."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    print(f"\nMerging adapter into {base_id} -> {out_dir} (CPU, {dtype}) ...")
    base = load_model(base_id, load_in_4bit=False, dtype=dtype, device_map="cpu")
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_id).save_pretrained(out_dir)
    print(f"Merged model saved to {out_dir}")

def push_to_hub(repo_id: str, folder: str, private: bool) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("Install huggingface_hub: pip install huggingface_hub")

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=folder, repo_type="model")
    print(f"Pushed {folder} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()