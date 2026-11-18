# The showcase image without the base checkpoint: everything else is the same (the environment, the ECB
# and opus-100 data, sft/, colab/translation_sft.py, the benchmark). At start-up the entrypoint
# downloads the checkpoint from Hugging Face (CKPT_REPO) into /opt/ckpt/ovhaillm-rl, the path the
# configs and scripts already use, unless a checkpoint is mounted there. Then it runs the command.
#
#   images/build_tutorial_hf.sh
#
# Built from the showcase image (images/tutorial/Dockerfile), with its filesystem minus /opt/ckpt
# copied into an empty image in the same layers, all under 10 GB (GitHub's registry limit).
ARG SRC=registry.gra.ai.cloud.ovh.net/4bebe832-6c48-4ce8-876e-f5e03dc6d740/ovhaillm-mt-tutorial:20261112-ecb
ARG SP=/opt/train-env/env/.venv/lib/python3.12/site-packages

FROM ${SRC} AS src
FROM src AS rest
RUN ["/bin/rm", "-rf", "/opt/ckpt", "/opt/train-env/env/.venv", "/usr/local/cuda-13.2"]
FROM src AS venv
RUN ["/bin/rm", "-rf", "/opt/train-env/env/.venv/lib/python3.12/site-packages/nvidia", \
     "/opt/train-env/env/.venv/lib/python3.12/site-packages/torch", \
     "/opt/train-env/env/.venv/lib/python3.12/site-packages/triton", \
     "/opt/train-env/env/.venv/lib/python3.12/site-packages/flash_attn_3", \
     "/opt/train-env/env/.venv/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"]

FROM scratch
ARG SP
COPY --from=rest / /
COPY --from=src /usr/local/cuda-13.2 /usr/local/cuda-13.2
COPY --from=venv /opt/train-env/env/.venv /opt/train-env/env/.venv
COPY --from=src ${SP}/nvidia ${SP}/nvidia
COPY --from=src ${SP}/torch ${SP}/torch
COPY --from=src ${SP}/triton ${SP}/triton
COPY --from=src ${SP}/flash_attn_3 ${SP}/flash_attn_3
COPY --from=src ${SP}/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so ${SP}/
# The download target, writable by OVH's uid 42420.
COPY --chmod=0755 images/tutorial-hf/fetch_ckpt.sh /opt/mt/fetch_ckpt.sh
RUN ["/bin/mkdir", "-m", "0777", "-p", "/opt/ckpt/ovhaillm-rl"]

# The showcase image's environment (copied from it: FROM scratch starts with none), plus CKPT_REPO:
# the public base checkpoint on Hugging Face.
ENV PATH=/opt/mt/sft:/opt/train-env/env/.venv/bin:/usr/local/cuda-12.8/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin:/opt/tensorrt/bin \
    VIRTUAL_ENV=/opt/train-env/env/.venv \
    TRAIN_ENV_HOME=/opt/train-env/env \
    LD_LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    CUDA_HOME=/usr/local/cuda-12.8 \
    _CUDA_COMPAT_PATH=/usr/local/cuda/compat \
    OPAL_PREFIX=/usr/local/mpi \
    OMPI_MCA_coll_hcoll_enable=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    NVIDIA_REQUIRE_CUDA="cuda>=9.0" \
    ENV=/etc/shinit_v2 \
    BASH_ENV=/etc/bash.bashrc \
    SHELL=/bin/bash \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX" \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 \
    TORCH_NCCL_USE_COMM_NONBLOCKING=0 \
    TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION=0 \
    CUDA_MODULE_LOADING=LAZY \
    CUDA_BINARY_LOADER_THREAD_COUNT=8 \
    UCC_CL_BASIC_TLS=^sharp \
    UCC_EC_CUDA_EXEC_NUM_THREADS=256 \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TRITON_CUOBJDUMP_PATH=/usr/local/cuda/bin/cuobjdump \
    TRITON_NVDISASM_PATH=/usr/local/cuda/bin/nvdisasm \
    TRITON_CUDACRT_PATH=/usr/local/cuda/include \
    TRITON_CUDART_PATH=/usr/local/cuda/include \
    TRITON_CUPTI_LIB_PATH=/usr/local/cuda/lib64 \
    TRITON_CUPTI_INCLUDE_PATH=/usr/local/cuda/include \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface \
    TILELANG_CACHE_DIR=/cache/tilelang \
    TRITON_CACHE_DIR=/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    HOME=/workspace \
    USER=ovh \
    DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    SFT_CONFIG=/opt/mt/sft/configs/mt_lora_ecb.yaml \
    BENCH_FILE=/opt/mt/data/open/ecb_enfr_eval.jsonl \
    MODEL_DIR=/opt/ckpt/ovhaillm-rl \
    OPEN_DATA_DIR=/opt/mt/data/open \
    USE_REENTRANT=1 \
    CKPT_REPO=OVHaiLLM/Olala-v1.0-mini-7A1B
WORKDIR /workspace
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "/opt/mt/fetch_ckpt.sh"]
# Same as the showcase: translation_sft.py (TRL) by default; `run_sft_bench.sh` for the custom trainer.
CMD ["python", "/opt/mt/collab/translation_sft.py", "--output-dir", "/workspace/out"]
