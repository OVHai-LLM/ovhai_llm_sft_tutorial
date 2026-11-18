#!/usr/bin/env bash
# Launch LoRA SFT with the custom trainer (sft/) on OVHcloud AI Training, with the showcase image
# (ovhaillm/olala-sft-tutorial): the open ECB en<->fr data is
# inside it and the base checkpoint is downloaded at start-up, so the only volume is the output bucket. The job benchmarks the base (BLEU / chrF++), trains, merges, benchmarks
# the merged model and prints both (sft/run_sft_bench.sh).
#
#   ovh/launch_sft_baked.sh                                   # mt_lora_ecb.yaml as is
#   ovh/launch_sft_baked.sh lora.r=16 training.num_train_epochs=1
#   BENCH_N=50 ovh/launch_sft_baked.sh training.max_steps=20  # quick check
#
# In the image:
#   /opt/ckpt/ovhaillm-rl          the base checkpoint (model_path), downloaded at start-up
#   /opt/mt/data/open            ecb_enfr_train.jsonl (4000), ecb_enfr_eval.jsonl (400, the benchmark),
#                                      opus100_enfr_*.jsonl (general domain)
#   /workspace/out -> $OUT_BUCKET/$NAME   bench_base.json, bench_finetuned.json, bench.txt,
#                                         adapter/, merged/, checkpoints/, sft_config.yaml
#
# Prerequisites (once): `ovhai login`. The image is public on Docker Hub (ovhaillm/olala-sft-tutorial).
# The job downloads the base checkpoint from Hugging Face at start-up.
#
# SSH_KEY=~/.ssh/id_ed25519.pub enables SSH into the job (for scp / rsync of the outputs; see
# TUTORIAL_SFT.md). It can only be set at launch.
# KEEP_ALIVE=1 keeps the job running after training (log in /workspace/out/run.log), so you can
# copy the outputs over SSH; stop it yourself with `ovhai job stop <id>`, or TIMEOUT stops it.
set -euo pipefail

IMAGE=${IMAGE:-ovhaillm/olala-sft-tutorial:20261117}
FLAVOR=${FLAVOR:-h100-1-gpu}          # `ovhai capabilities flavor list`
GPU=${GPU:-1}                         # train.sh runs one DDP rank per GPU
REGION=${REGION:-GRA}
OUT_BUCKET=${OUT_BUCKET:-ovhaillm-mt-sft-out}
CONFIG=${CONFIG:-/opt/mt/sft/configs/mt_lora_ecb.yaml}
BENCH_N=${BENCH_N:-}                  # empty = all 400 eval sentences
NAME=${NAME:-ovhaillm-mt-sft-$(date +%Y%m%d-%H%M)}
TIMEOUT=${TIMEOUT:-8h}                # always with a unit: a bare number is seconds
CMD=(run_sft_bench.sh "$@")
[ "${KEEP_ALIVE:-0}" = 1 ] && CMD=(bash -c '"$@" 2>&1 | tee /workspace/out/run.log; sleep infinity' _ "${CMD[@]}")

set -x
ovhai job run "$IMAGE" \
    --name "$NAME" \
    --flavor "$FLAVOR" \
    --gpu "$GPU" \
    --timeout "$TIMEOUT" \
    --volume "$OUT_BUCKET@$REGION/$NAME:/workspace/out:rw" \
    ${SSH_KEY:+--ssh-public-keys "$SSH_KEY"} \
    --env HF_HUB_OFFLINE=1 \
    --env CONFIG="$CONFIG" \
    --env BENCH_N="$BENCH_N" \
    --env OUT=/workspace/out \
    -- "${CMD[@]}"
