#!/usr/bin/env bash
# Build (and optionally push) the olala-mt trainer image.
#
#   images/build.sh                                   # local: olala-mt-trainer:<tag> and :latest
#   REGISTRY=git.corp.linguacustodia.com:5050/<group>/olala-mt images/build.sh   # + push
#
# Needs read access to the base, gcaillaut/olala-env on the corp registry
# (`docker login git.corp.linguacustodia.com:5050`). It is a ~30 GB pull.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME=olala-mt-trainer
TAG=${TAG:-$(date +%Y%m%d-%H%M)}
IMAGE=${REGISTRY:+$REGISTRY/}$NAME

docker build \
    --platform linux/amd64 \
    ${BASE:+--build-arg BASE=$BASE} \
    -f "$ROOT/images/trainer/Dockerfile" \
    -t "$IMAGE:$TAG" -t "$IMAGE:latest" \
    "$ROOT"

if [ -n "${REGISTRY:-}" ] && [ "${PUSH:-1}" = "1" ]; then
    docker push "$IMAGE:$TAG"
    docker push "$IMAGE:latest"
    echo ">> pushed $IMAGE:$TAG"
else
    echo ">> built $IMAGE:$TAG (not pushed; set REGISTRY to push)"
fi
