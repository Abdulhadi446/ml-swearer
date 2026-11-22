#!/usr/bin/env bash
# train.sh — download/download the Qwen3.5-0.8B-Base model, run QLoRA fine-tuning,
# and save the final merged model to /kaggle/working/.
#
# Usage:
#   ./train.sh [OPTIONS]
#
# Options (all optional):
#   --skip-deps      skip pip install (requires transformers already present)
#   --no-4bit        disable 4-bit NF4 QLoRA (plain fp16/bf16 LoRA)
#   --epochs N       number of epochs (default 3)
#   --batch-size N   per-device batch size (default 8)
#   --lr L           learning rate (default 2e-4)
#   --max-seq-len N  max sequence length for generation (default 256)
#   --warmup-ratio R warmup fraction (default 0.03)
#   --weight-decay D weight decay (default 0.01)
#   --seed S         random seed (default 42)
set -euo pipefail

REPO_ROOT="/home/abdulhadi/Projects/ml-spearer"
# Allow overriding the working directory
WORK_DIR="${WORK_DIR:-/kaggle/working}"
# If /kaggle/working is not writable (e.g. no Kaggle setup), fall back
if [ ! -d "$(dirname "$WORK_DIR")" ] || [ ! -w "$(dirname "$WORK_DIR")" ]; then
    WORK_DIR="/tmp/kaggle/working"
    echo "==> Cannot create /kaggle/working, falling back to: $WORK_DIR"
fi
mkdir -p "$WORK_DIR/model" "$WORK_DIR/model-merged"

# ---------------------------------------------------------------------------
# 1. Install dependencies (only if pip is available and not skipped)
# ---------------------------------------------------------------------------
if [[ ! -f requirements.txt ]]; then
    echo "==> Installing requirements ..."
    pip install -q -r requirements.txt
else
    echo "==> Requirements already installed (skipping pip)."
fi

# If running on TPU, ensure torch_xla is available (Kaggle TPU VMs may need it)
if python3 -c "import torch; print(getattr(torch, 'xla_available', False))" 2>/dev/null | grep -q "True"; then
    echo "==> TPU detected — ensuring torch_xla is installed ..."
    pip install -q --upgrade torch_xla 2>&1 | tail -1
fi

# ---------------------------------------------------------------------------
# 2. Validate the dataset (no torch required)
# ---------------------------------------------------------------------------
echo "==> Checking dataset integrity ..."
python3 train_qlora.py --check-data

# ---------------------------------------------------------------------------
# 3. Train LoRA adapter (Qwen3.5-0.8B-Base)
#    The script auto-detects TPU vs GPU vs CPU
# ---------------------------------------------------------------------------
echo "==> Training LoRA adapter ..."
python3 train_qlora.py \
    --train-file dataset/combined.jsonl \
    --val-file dataset/validation.jsonl \
    --output-dir "$WORK_DIR/model-lora" \
    --prompt-format plain \
    --epochs 3 \
    --batch-size 8 \
    --lr 2e-4 \
    --max-seq-len 256 \
    --warmup-ratio 0.03 \
    --weight-decay 0.01 \
    --seed 42

# ---------------------------------------------------------------------------
# 4. Merge adapter into a fresh fp16/bf16 copy of the base model
#    -> single uploadable model folder
# ---------------------------------------------------------------------------
echo "==> Merging adapter into base model (single model) ..."
python3 train_qlora.py \
    --model Qwen/Qwen3.5-0.8B-Base \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --output-dir "$WORK_DIR/model-merged" \
    --prompt-format plain \
    --epochs 3 \
    --batch-size 8 \
    --lr 2e-4 \
    --merge-and-save \
    --merge-dir "$WORK_DIR/model-merged" \
    --seed 42

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
echo ""
echo "==> Final merged model saved to: $WORK_DIR/model-merged/"
ls -lh "$WORK_DIR/model-merged/"