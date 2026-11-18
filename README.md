# olala-mt-tuto

LoRA SFT for **Olala 7A1B** (hybrid Mamba-3 / attention, MoE) on **H100**: a Docker image, training code, and an English↔French translation tutorial. Everything is set up to run on OVHcloud AI Training.

**Repository:** [github.com/OVHai-LLM/ovhai_llm_sft_tutorial](https://github.com/OVHai-LLM/ovhai_llm_sft_tutorial). 

**Related repositories:** [`OVHai-LLM/olala-v1`](https://github.com/OVHai-LLM/olala-v1) (model code, checkpoint conversion, training fixes), [`OVHai-LLM/mamba`](https://github.com/OVHai-LLM/mamba) (the Mamba-3 kernels) and [`OVHai-LLM/vllm`](https://github.com/OVHai-LLM/vllm) (the vLLM fork, for serving).

**Quick start:** [`TUTORIAL_SFT.md`](TUTORIAL_SFT.md) is a detailed guide to the in-domain SFT tutorial.

## Image

One public image on Docker Hub, [`ovhaillm/olala-sft-tutorial`](https://hub.docker.com/r/ovhaillm/olala-sft-tutorial): run it on OVH or anywhere else, no login needed. `20261117` and `latest` are the same image.

It holds the environment, the open ECB en↔fr data, both trainers and the benchmark (24.3 GB), but not the checkpoint, which it downloads at start-up. It runs the same experiment in two versions: score the base model, LoRA fine-tune it, score it again, compare.
- **The teaching version (the default command):** `colab/translation_sft.py`, a short script with Hugging Face TRL's `SFTTrainer`, the same code as the Colab notebook. LoRA r16, 1 epoch, lr 1e-4. It saves the LoRA adapter and prints the before / after table.
- **The full version (`run_sft_bench.sh`):** our own trainer in `sft/`. LoRA r32, 2 epochs, lr 5e-5, from a YAML config, with any setting changeable on the command line (`lora.r=16`). It also writes the merged model (ready to serve) and every translation as JSON. The reference results in `TUTORIAL_SFT.md` come from this one.

**The checkpoint download.** The entrypoint downloads the base checkpoint from Hugging Face into `/opt/ckpt/ovhaillm-rl`, the path the scripts use, then runs the command.
- **`CKPT_REPO`:** the Hugging Face repo to download, by default `OVHaiLLM/Olala-v1.0-mini-7A1B`. `CKPT_REV` picks a branch, tag or commit; `HF_TOKEN` is needed only while the repo is private.
- **Internet access** is needed at start-up for the download (about 14 GB). The rest runs offline.
- **No download** if a checkpoint (`config.json` + `*.safetensors`) is already mounted at `/opt/ckpt/ovhaillm-rl`, e.g. `-v /path/to/checkpoint:/opt/ckpt/ovhaillm-rl:ro`.

```bash
docker run --rm --gpus '"device=0"' --ipc=host -v $PWD/out:/workspace/out \
    ovhaillm/olala-sft-tutorial:20261117                    # downloads, then the teaching version
docker run ... ovhaillm/olala-sft-tutorial:20261117 run_sft_bench.sh    # or the full version
```


## Repository layout

```
colab/    the tutorial: translation_sft.py and the Colab notebook (built by build_notebook.py); data in open_data/
sft/      generic LoRA SFT: sft_lora.py, merge_lora.py, train.sh, configs/
images/   Dockerfiles (trainer/, tutorial/, sft/) and their build scripts
open_data/ open en<->fr SFT sets and the scripts that build them: ECB finance (4000 + 400, make_ecb_enfr.py)
          and opus-100 general (2000 + 200, make_opus100_enfr.py)
ovh/      OVHcloud AI Training launch scripts (see ovh/README.md)
```





