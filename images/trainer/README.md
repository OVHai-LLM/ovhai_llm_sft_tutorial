# olala-mt-trainer

LoRA SFT for Olala checkpoints on H100. It's built on `olala-env:505ff88`, which already contains the torch / mamba3 / FA3 stack compiled for `sm_90a`. The code it runs lives in `../../sft/`.

## Build

```bash
docker login git.corp.linguacustodia.com:5050     # needs read access to gcaillaut/olala-env
images/build.sh                                    # -> olala-mt-trainer:<date> and :latest
REGISTRY=git.corp.linguacustodia.com:5050/<group>/olala-mt images/build.sh   # build + push
```

## Run on this box

Checkpoints and data are mounted at run time, not baked into the image:

```bash
docker run --rm -it --gpus all --ipc=host \
    -v /data/home/jingshu.liu/olala-mt:/workspace/olala-mt \
    olala-mt-trainer:latest \
    train.sh /workspace/olala-mt/sft/configs/mt_lora.yaml
```

- **Smoke test first:** append `NPROC=1` via `-e NPROC=1` and add `training.max_steps=20 merge_when_done=false` after the config path.
- **Override any key** with `key=value`, for example `lora.r=16 training.learning_rate=5e-5 output_dir=/workspace/olala-mt/runs/x`.
- **Check the attention backend:** the log should print `Using attention implementation: fa3`. If it says `fa2`, the base image is wrong.
- **Iterate on `sft/` without rebuilding:** add `-v /data/home/jingshu.liu/olala-mt/sft:/opt/olala-mt/sft` to the mount list.

## Output

```
<output_dir>/adapter/      LoRA adapter + tokenizer
<output_dir>/merged/       full checkpoint (only targeted weights changed), servable by olala-vllm
<output_dir>/checkpoints/  Trainer checkpoints; a rerun resumes from the newest
```

To merge an adapter by hand: `python /opt/olala-mt/sft/merge_lora.py BASE ADAPTER OUT`.
