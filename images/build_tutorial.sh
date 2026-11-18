#!/usr/bin/env bash
# Build (and optionally push) the tutorial image: the clean sft image + colab/translation_sft.py.
#
#   images/build_tutorial.sh
#   PUSH=1 images/build_tutorial.sh          # + push to OVH's project registry
#
# Then checks that the base model's code name appears nowhere outside the checkpoint's model files:
# image config, history, file paths and file contents. The base image is images/sft/Dockerfile's.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REGISTRY=${REGISTRY:-registry.gra.ai.cloud.ovh.net/4bebe832-6c48-4ce8-876e-f5e03dc6d740}
NAME=ovhaillm-mt-tutorial
TAG=${TAG:-$(date +%Y%m%d-%H%M)}
IMAGE=$REGISTRY/$NAME
CODE_NAME=${CODE_NAME:-olala}
KEEP='^/opt/ckpt/ovhaillm-rl/(modeling_'$CODE_NAME'\.py|configuration_'$CODE_NAME'\.py|config\.json|chat_template\.jinja)$'

docker build --platform linux/amd64 ${BASE:+--build-arg BASE=$BASE} \
    -f "$ROOT/images/tutorial/Dockerfile" -t "$IMAGE:$TAG" "$ROOT"

echo ">> scanning $IMAGE:$TAG"
n=$( { docker image inspect "$IMAGE:$TAG" --format '{{json .Config}}'; \
       docker history --no-trunc --format '{{.CreatedBy}} {{.Comment}}' "$IMAGE:$TAG"; } | grep -ci "$CODE_NAME" || true)
[ "$n" = 0 ] || { echo "code name in the image config or history ($n lines)"; exit 1; }
left=$(docker run --rm --network none --entrypoint bash "$IMAGE:$TAG" -c \
    "{ find / -xdev -iname '*$CODE_NAME*'; grep -rli -a $CODE_NAME / --exclude-dir={proc,sys,dev} 2>/dev/null; } | sort -u" \
    | grep -v -E "$KEEP" || true)
[ -z "$left" ] || { echo "code name still in:"; echo "$left"; exit 1; }
echo ">> clean: no mention outside the checkpoint's model files"

if [ "${PUSH:-0}" = "1" ]; then
    docker push "$IMAGE:$TAG"
    echo ">> pushed $IMAGE:$TAG"
else
    echo ">> built $IMAGE:$TAG (not pushed; PUSH=1 to push)"
fi
