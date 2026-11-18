# The translation SFT tutorial (colab/translation_sft.py: TRL SFTTrainer + LoRA), ready to run
# on OVHcloud AI Training: the ECB en<->fr LoRA SFT showcase.
#
#   images/build_tutorial.sh
#
# Built on the clean sft image (images/sft/Dockerfile), which already holds the
# training env (with TRL), the base checkpoint (/opt/ckpt/ovhaillm-rl) and the ECB train / eval
# data (/opt/mt/data/open), and in which the base model's code name appears only inside the
# checkpoint's model files. This adds the script; images/build_tutorial.sh scans the result.
# (No check in a RUN step here: RUN commands are recorded in the image history.)
ARG BASE=registry.gra.ai.cloud.ovh.net/4bebe832-6c48-4ce8-876e-f5e03dc6d740/ovhaillm-mt-sft:20261112-ecb
FROM ${BASE}

COPY --chmod=0755 colab/translation_sft.py /opt/mt/collab/
RUN /opt/train-env/env/.venv/bin/python -m compileall -q /opt/mt/collab

# MODEL_DIR: the checkpoint. OPEN_DATA_DIR: where the script finds ecb_enfr_{train,eval}.jsonl.
# USE_REENTRANT=1: the image's upstream mamba_ssm needs reentrant gradient checkpointing.
ENV MODEL_DIR=/opt/ckpt/ovhaillm-rl \
    OPEN_DATA_DIR=/opt/mt/data/open \
    USE_REENTRANT=1 \
    HF_DATASETS_OFFLINE=1
WORKDIR /workspace
# Evaluate the base on the 400 ECB eval pairs, LoRA-SFT on the 4000 train pairs, evaluate again,
# print the table and save the adapter to /workspace/out/lora (mount the output bucket there).
CMD ["python", "/opt/mt/collab/translation_sft.py", "--output-dir", "/workspace/out"]
