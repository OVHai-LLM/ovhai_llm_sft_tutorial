# olala-mt LoRA SFT trainer, for H100 (sm_90a).
#
#   images/build.sh                      # builds olala-mt-trainer:<tag>, context = olala-mt root
#
# BASE: Gaetan's olala-env, the same image olala-rlhf's DPO trainer and
# olala-antidoom train on. It already holds everything this needs and is already
# compiled for Hopper: torch 2.11.0+cu128, transformers, peft, accelerate, trl,
# datasets, tensorboard, bitsandbytes, scattermoe, mamba_ssm with the tilelang
# mamba3 kernels (JIT, so it carries a CUDA 12.8 toolkit for nvcc at runtime),
# and FlashAttention 3 built for sm_90a -- which modeling_olala.py prefers over
# FA2 at import time, silently falling back if it is missing.
#
# PINNED TO A SHA, never :latest: the short SHA identifies the whole frozen env.
#
# NOTHING IS pip-INSTALLED HERE. olala-env's venv is a frozen snapshot that is
# deliberately not re-resolvable; any resolver run can swap torch for a PyPI
# build that does not know the mamba3 kernels. sft/ is plain scripts that only
# import what the base already has, and the check below fails the build if not.
# If something genuinely is missing: `uv pip install --no-deps <pkg>`, never
# without --no-deps.
ARG BASE=git.corp.linguacustodia.com:5050/gcaillaut/olala-env/olala:505ff88
FROM ${BASE}

ENV OLALA_VENV=/opt/olala-env/env/.venv \
    PYTHONUNBUFFERED=1

WORKDIR /opt/olala-mt
COPY sft ./sft
RUN chmod 0755 sft/train.sh \
    && ${OLALA_VENV}/bin/python -c "\
import torch, transformers, peft, accelerate, datasets, yaml, safetensors, tensorboard; \
import scattermoe, mamba_ssm, tilelang; \
import flash_attn_interface; \
archs = torch._C._cuda_getArchFlags().split(); \
assert 'sm_90' in archs, f'torch has no sm_90 kernels: {archs}'; \
print('torch', torch.__version__, '| transformers', transformers.__version__, \
      '| peft', peft.__version__, '| archs', archs, '| fa3 present')" \
    && ${OLALA_VENV}/bin/python sft/sft_lora.py --help > /dev/null \
    && ${OLALA_VENV}/bin/python -c "import sys; sys.path.insert(0, 'sft'); import merge_lora, sft_lora"

ENV PATH=/opt/olala-mt/sft:${PATH}

# OVHcloud AI Training runs the container as uid 42420, not root. The import
# check above created /cache/tilelang and /cache/inductor as root 0755, and the
# mamba3 JIT must write its compiled kernels there on the first forward pass.
RUN mkdir -p /cache/triton /cache/huggingface && chmod -R 0777 /cache

# ...and that uid must exist in /etc/passwd. torch.compile names its cache dir after
# getpass.getuser(), which raises "getpwuid(): uid not found: 42420" otherwise -- found by
# running the tutorial as 42420. USER covers any other uid: getuser() reads it first.
RUN (getent group 42420 || groupadd -g 42420 ovh) \
    && (getent passwd 42420 || useradd -u 42420 -g 42420 -M -d /workspace -s /bin/bash ovh)
ENV USER=ovh

# Cache variables are deliberately left as the base sets them (/cache): the
# mamba3 JIT output is device-specific, so it belongs to the container, not a
# shared mount.
ENV HOME=/workspace
WORKDIR /workspace

# NGC's entrypoint checks the host driver against the image's CUDA, then exec "$@".
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["bash"]
