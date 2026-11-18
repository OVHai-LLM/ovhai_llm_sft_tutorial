#!/usr/bin/env bash
# Build step 1 of the showcase image (step 2: images/build_tutorial.sh). Users run the showcase image.
# Build (and optionally push) the self-contained SFT image: trainer image + the base checkpoint
# + the open en<->fr sets in open_data/ (ECB finance, opus-100 general; no in-house data).
#
#   CKPT=/path/to/checkpoint images/build_sft.sh
#   CKPT=... REGISTRY=registry.gra.ai.cloud.ovh.net/<project> images/build_sft.sh   # + push
#
# The checkpoint is passed as a separate build context; open_data/ comes from the main context
# (the olala-mt root), whose .dockerignore keeps checkpoints/ and the in-house data out.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# CKPT: the base checkpoint folder, OvhaiLLM-RL (RL run ART, global step 294; on Hugging Face,
# OVHaiLLM/olala-rl, private). See TUTORIAL_SFT.md for its sha256.
CKPT=${CKPT:?set CKPT to the base checkpoint folder}
[ -f "$CKPT/model.safetensors" ] || { echo "CKPT=$CKPT has no model.safetensors"; exit 1; }
NAME=ovhaillm-mt-sft
TAG=${TAG:-$(date +%Y%m%d-%H%M)}
IMAGE=${REGISTRY:+$REGISTRY/}$NAME

docker build \
    --platform linux/amd64 \
    ${BASE:+--build-arg BASE=$BASE} \
    --build-context ckpt="$CKPT" \
    -f "$ROOT/images/sft/Dockerfile" \
    -t "$IMAGE:$TAG" -t "$IMAGE:latest" \
    "$ROOT"

if [ -n "${REGISTRY:-}" ] && [ "${PUSH:-1}" = "1" ]; then
    docker push "$IMAGE:$TAG"
    docker push "$IMAGE:latest"
    echo ">> pushed $IMAGE:$TAG"
else
    echo ">> built $IMAGE:$TAG (not pushed; set REGISTRY to push)"
fi
