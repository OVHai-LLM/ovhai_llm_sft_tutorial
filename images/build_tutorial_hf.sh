#!/usr/bin/env bash
# Build (and optionally push) the showcase image without the checkpoint (images/tutorial-hf/Dockerfile).
# It downloads the checkpoint from Hugging Face (CKPT_REPO) when a container starts.
#
#   images/build_tutorial_hf.sh
#   REGISTRY=ghcr.io/<user> PUSH=1 images/build_tutorial_hf.sh      # + push
#
# SRC picks the showcase image to start from (default in the Dockerfile).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME=${NAME:-ovhaillm-mt-tutorial}
TAG=${TAG:-$(date +%Y%m%d)-hf}
IMAGE=${REGISTRY:+$REGISTRY/}$NAME

docker build \
    --platform linux/amd64 \
    ${SRC:+--build-arg SRC=$SRC} \
    -f "$ROOT/images/tutorial-hf/Dockerfile" \
    -t "$IMAGE:$TAG" \
    "$ROOT"

if [ -n "${REGISTRY:-}" ] && [ "${PUSH:-0}" = "1" ]; then
    docker push "$IMAGE:$TAG"
    echo ">> pushed $IMAGE:$TAG"
else
    echo ">> built $IMAGE:$TAG (not pushed; REGISTRY=... PUSH=1 to push)"
fi
