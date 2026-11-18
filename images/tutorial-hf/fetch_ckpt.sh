#!/usr/bin/env bash
# Entrypoint of the image without the checkpoint: make sure the base checkpoint is in $MODEL_DIR,
# then run the command. If $MODEL_DIR already holds config.json and *.safetensors (a mounted volume, or an
# earlier download in the same container), nothing is downloaded.
#
#   CKPT_REPO   the Hugging Face model repo to download (default in the image: OVHaiLLM/Olala-v1.0-mini-7A1B)
#   CKPT_REV    optional: a branch, tag or commit of that repo
#   HF_TOKEN    optional: only needed while the repo is private
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/opt/ckpt/ovhaillm-rl}
if [ ! -f "$MODEL_DIR/config.json" ] || ! compgen -G "$MODEL_DIR/*.safetensors" > /dev/null; then
    if [ -z "${CKPT_REPO:-}" ]; then
        echo ">> no checkpoint in $MODEL_DIR and CKPT_REPO is not set: set CKPT_REPO=<org/name>, or mount the checkpoint folder at $MODEL_DIR" >&2
        exit 1
    fi
    echo ">> downloading $CKPT_REPO${CKPT_REV:+@$CKPT_REV} to $MODEL_DIR (about 14 GB)"
    HF_HUB_OFFLINE=0 python - "$CKPT_REPO" "$MODEL_DIR" "${CKPT_REV:-}" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, local_dir, rev = sys.argv[1], sys.argv[2], sys.argv[3] or None
snapshot_download(repo, local_dir=local_dir, revision=rev,
                  allow_patterns=["*.json", "*.py", "*.jinja", "*.safetensors", "*.txt"])
EOF
    rm -rf "$MODEL_DIR/.cache"
    echo ">> checkpoint ready in $MODEL_DIR"
fi
exec "$@"
