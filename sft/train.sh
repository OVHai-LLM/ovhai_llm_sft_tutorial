#!/usr/bin/env bash
# LoRA SFT on every GPU of this node (DDP via torchrun).
#
#   sft/train.sh sft/configs/mt_lora.yaml [key=value ...]
#   NPROC=1 sft/train.sh sft/configs/mt_lora.yaml training.max_steps=20   # smoke test
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG=${1:?usage: sft/train.sh CONFIG [key=value ...]}
shift
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}
exec torchrun --standalone --nproc_per_node="$NPROC" "$HERE/sft_lora.py" --config "$CONFIG" "$@"
