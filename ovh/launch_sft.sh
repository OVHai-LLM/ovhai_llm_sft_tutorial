#!/usr/bin/env bash
# Launch LoRA SFT (sft/sft_lora.py) on OVHcloud AI Training with the trainer image.
#
#   CKPT_BUCKET=olala-ckpt DATA_BUCKET=olala-mt-data TRAIN_FILES='[/workspace/data/train.jsonl]' \
#       ovh/launch_sft.sh [extra key=value overrides ...]
#
# Unlike the tutorial image, the trainer image holds no checkpoint and no data: they come from
# Object Storage buckets mounted read-only, and the output bucket is mounted read-write.
#   /workspace/ckpt   <- $CKPT_BUCKET   (an HF-format Olala checkpoint at the bucket root)
#   /workspace/data   <- $DATA_BUCKET
#   /workspace/out    -> $OUT_BUCKET/$NAME   (adapter/, merged/, checkpoints/, sft_config.yaml)
#
# Extra arguments are passed to sft_lora.py as overrides, e.g.
#   completion_field=chosen lora.r=16 training.num_train_epochs=1
set -euo pipefail

IMAGE=${IMAGE:-git.corp.linguacustodia.com:5050/jingshu/olala-mt-tuto/olala-mt-trainer:20261107-0703}
FLAVOR=${FLAVOR:-h100-1-gpu}          # `ovhai capabilities flavor list`
GPU=${GPU:-1}                         # train.sh runs one DDP rank per GPU
REGION=${REGION:-GRA}
CKPT_BUCKET=${CKPT_BUCKET:?set CKPT_BUCKET to the bucket holding the base checkpoint}
DATA_BUCKET=${DATA_BUCKET:?set DATA_BUCKET to the bucket holding the training data}
TRAIN_FILES=${TRAIN_FILES:?set TRAIN_FILES, e.g. "[/workspace/data/train.jsonl]"}
OUT_BUCKET=${OUT_BUCKET:-ovhaillm-mt-sft-out}
CONFIG=${CONFIG:-/opt/olala-mt/sft/configs/mt_lora.yaml}
NAME=${NAME:-ovhaillm-mt-sft-$(date +%Y%m%d-%H%M)}

set -x
ovhai job run "$IMAGE" \
    --name "$NAME" \
    --flavor "$FLAVOR" \
    --gpu "$GPU" \
    --volume "$CKPT_BUCKET@$REGION/:/workspace/ckpt:ro" \
    --volume "$DATA_BUCKET@$REGION/:/workspace/data:ro" \
    --volume "$OUT_BUCKET@$REGION/$NAME:/workspace/out:rw" \
    --env HF_HUB_OFFLINE=1 \
    -- train.sh "$CONFIG" \
        model_path=/workspace/ckpt \
        "train_files=$TRAIN_FILES" \
        output_dir=/workspace/out \
        "$@"
