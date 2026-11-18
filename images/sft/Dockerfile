# Step 1 of the showcase image (ovhaillm-mt-tutorial; images/tutorial/Dockerfile adds the TRL script on top).
# In-domain LoRA SFT, self-contained for OVHcloud AI Training: the training env plus the base
# checkpoint, small OPEN datasets and a BLEU / chrF++ benchmark, so the job only needs an output
# bucket. No in-house data goes into this image.
#
#   images/build_sft.sh        # passes the checkpoint dir as a build context
#
#   /opt/ckpt/ovhaillm-rl                OvhaiLLM-RL (RL run ART, global step 294)
#   /opt/mt/data/open                    open_data/:
#       ecb_enfr_{train,eval}.jsonl        OPUS ECB (central banking / finance) en<->fr, 4000 + 400
#                                          pairs: the training set and the default benchmark
#       opus100_enfr_{train,eval}.jsonl    opus-100 general en<->fr, 2000 + 200: a general-domain check
#   /opt/mt/sft                          sft/: training, merge, benchmark; configs/mt_lora_ecb.yaml
#                                        is the default config (mt_lora_baked.yaml: same on opus-100)
#   /opt/train-env/env/.venv             the Python env
#
# The base model's internal code name is kept only inside the checkpoint folder: its modeling /
# configuration Python files, the config.json that points at them (transformers needs it to load
# the model) and the chat template. Everything else is renamed or removed in the `clean` stage,
# and the final stage copies the cleaned filesystem into an empty image, in several layers, so no
# earlier layer, label, env var or build step of the base image survives. The build fails if a
# mention is left.
ARG BASE=git.corp.linguacustodia.com:5050/jingshu/olala-mt-tuto/olala-mt-trainer:20261107-0703

FROM ${BASE} AS clean
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG OLD=/opt/olala-env
ARG NEW=/opt/train-env
ARG SP=/opt/train-env/env/.venv/lib/python3.12/site-packages

# The training env, moved to a neutral path. Dropped: the env's own source tree and helper
# scripts, and the packages SFT doesn't use (vLLM, verl, transfer_queue, renderers).
RUN rm -rf /opt/olala-mt /opt/olala-ckpt /cache/uv /root/.cache \
 && mv ${OLD} ${NEW} \
 && cd ${NEW} && rm -rf PINS check_env.py serve.sh setup_*.sh verify.sh write_pins.sh env/renderers \
 && rm -rf env/olala env/scattermoe/scattermoe/*.bak-* \
 && rm -rf ${SP}/vllm ${SP}/vllm-*.dist-info ${SP}/verl ${SP}/verl-*.dist-info ${SP}/transfer_queue \
           ${SP}/renderers-*.dist-info ${SP}/_editable_impl_renderers.pth ${NEW}/env/.venv/bin/vllm \
 # ... and what only serving / RL use: FlashInfer, the MLA kernels, Ray, transfer_queue's and
 # rl_insight's top-level folders (tests, tutorial, scripts, recipe)
 && rm -rf ${SP}/flashinfer ${SP}/flashinfer_* ${SP}/tokenspeed_* ${SP}/ray ${SP}/ray-*.dist-info ${NEW}/env/.venv/bin/ray \
           ${SP}/transferqueue-*.dist-info ${SP}/rl_insight ${SP}/rl_insight-*.dist-info \
           ${SP}/tests ${SP}/tutorial ${SP}/scripts ${SP}/recipe \
 # The base image's own PyTorch stack (system Python packages, PyTorch sources, MKL): the env has
 # its own torch and doesn't see the system site-packages
 && rm -rf /usr/local/lib/python3.12 /opt/pytorch /usr/local/lib/libmkl_* \
 && { for f in /usr/local/bin/*; do if head -c 64 "$f" | grep -qa '^#!.*python'; then rm -f "$f"; fi; done; } \
 # paths in shebangs, .pth files, RECORDs, direct_url.json ...
 && { grep -rIlZ "${OLD}" ${NEW} || true; } | xargs -0 -r sed -i "s#${OLD}#${NEW}#g" \
 # ... and in compiled libraries (build paths in debug strings): same length, so the layout is unchanged
 && [ ${#OLD} -eq ${#NEW} ] \
 && { grep -rlZ -a "${OLD}" ${NEW} --exclude=*.pyc || true; } | xargs -0 -r perl -pi -e "s#\Q${OLD}\E#${NEW}#g" \
 # comments naming the model: scattermoe's dtype-fix notes, the import stub's docstring
 && sed -i 's/# OLALA fix:/# dtype fix:/' ${NEW}/env/scattermoe/scattermoe/parallel_experts.py \
 && sed -i -E 's/\b[Oo]lala([- ]inference)/model\1/g; s/\b[Oo]lala.s vLLM path/the vLLM path/g; s/\b[Oo]lala inference/inference/g' \
        ${SP}/selective_scan_cuda.py \
 # bytecode embeds source paths: drop it, recompiled below
 && find / -xdev -name __pycache__ -type d -prune -exec rm -rf {} +

# sacrebleu for the benchmark (sft/bench_mt.py), with the three of its dependencies the env lacks.
# --no-deps: the env is a frozen snapshot.
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
RUN uv pip install --python ${NEW}/env/.venv/bin/python --no-deps --no-cache \
        sacrebleu==2.6.0 portalocker colorama lxml \
 && rm -f /usr/local/bin/uv

# The checkpoint: only what loading needs. --chmod makes it readable by OVH's uid 42420.
COPY --from=ckpt --chmod=0755 config.json generation_config.json configuration_olala.py modeling_olala.py \
     tokenizer.json tokenizer_config.json chat_template.jinja MERGED_FROM.txt model.safetensors \
     /opt/ckpt/ovhaillm-rl/
COPY --chmod=0755 open_data/make_ecb_enfr.py open_data/ecb_enfr_train.jsonl open_data/ecb_enfr_eval.jsonl \
     open_data/make_opus100_enfr.py open_data/opus100_enfr_train.jsonl open_data/opus100_enfr_eval.jsonl \
     /opt/mt/data/open/
COPY --chmod=0755 sft/sft_lora.py sft/merge_lora.py sft/bench_mt.py sft/train.sh sft/run_sft_bench.sh /opt/mt/sft/
COPY --chmod=0755 sft/configs/mt_lora_ecb.yaml sft/configs/mt_lora_baked.yaml /opt/mt/sft/configs/

# Recompile the bytecode (uid 42420 can't write it at run time), check the imports training needs
# (causal_conv1d and cut_cross_entropy fail to import in the base env too; the model falls back),
# then check: no mention of the
# code name anywhere, in paths or contents, outside the checkpoint's model files.
RUN ${NEW}/env/.venv/bin/python -m compileall -q -j 0 ${SP} ${NEW}/env /opt/mt/sft > /dev/null || true \
 && ${NEW}/env/.venv/bin/python -c "import sacrebleu, peft, transformers, datasets, mamba_ssm, fla, flash_attn, flash_attn_interface, scattermoe" \
 && KEEP='^/opt/ckpt/ovhaillm-rl/(modeling_olala\.py|configuration_olala\.py|config\.json|chat_template\.jinja)$' \
 && left=$( { find / -xdev -iname '*olala*'; grep -rli -a olala / --exclude-dir={proc,sys,dev} 2>/dev/null; } | sort -u | grep -v -E "$KEEP" || true) \
 && if [ -n "$left" ]; then echo "code name still in:"; echo "$left" | head -50; exit 1; fi \
 && echo "clean: no mention outside the checkpoint's model files"

# The final image: the cleaned filesystem copied into an empty image, in several layers of at most
# ~7.5 GB (plus the 13.6 GB checkpoint, a single file). One 54 GB layer failed to pull on OVH
# ("Error image pull" after ~9 minutes); the earlier multi-layer image of the same size pulled fine.
# The `rest` stage is the filesystem without the big folders, which get a layer each below.
FROM clean AS rest
RUN ["/bin/rm", "-rf", "/opt/ckpt", "/opt/train-env/env/.venv", "/usr/local/cuda-13.2"]
FROM clean AS venv
RUN ["/bin/rm", "-rf", "/opt/train-env/env/.venv/lib/python3.12/site-packages/nvidia", \
     "/opt/train-env/env/.venv/lib/python3.12/site-packages/torch", \
     "/opt/train-env/env/.venv/lib/python3.12/site-packages/triton", \
     "/opt/train-env/env/.venv/lib/python3.12/site-packages/flash_attn_3", \
     "/opt/train-env/env/.venv/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"]

FROM scratch
ARG SP=/opt/train-env/env/.venv/lib/python3.12/site-packages
COPY --from=rest / /
COPY --from=clean /usr/local/cuda-13.2 /usr/local/cuda-13.2
COPY --from=venv /opt/train-env/env/.venv /opt/train-env/env/.venv
COPY --from=clean ${SP}/nvidia ${SP}/nvidia
COPY --from=clean ${SP}/torch ${SP}/torch
COPY --from=clean ${SP}/triton ${SP}/triton
COPY --from=clean ${SP}/flash_attn_3 ${SP}/flash_attn_3
COPY --from=clean ${SP}/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so ${SP}/
COPY --from=clean /opt/ckpt /opt/ckpt
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
    SFT_CONFIG=/opt/mt/sft/configs/mt_lora_ecb.yaml \
    BENCH_FILE=/opt/mt/data/open/ecb_enfr_eval.jsonl
WORKDIR /workspace
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
# Benchmark the base, LoRA-SFT, benchmark the merged model, print both. Output in /workspace/out.
CMD ["run_sft_bench.sh"]
