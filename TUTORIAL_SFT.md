# Tutorial: in-domain LoRA SFT for translation

Fine-tune the base model on finance en↔fr data, and measure BLEU / chrF++ before and after.

**The data:** central banking and finance sentence pairs from the open [ECB corpus](https://opus.nlpl.eu/ECB-v1.php), in `open_data/`:
- `ecb_enfr_train.jsonl`: 4000 pairs, half in each direction;
- `ecb_enfr_eval.jsonl`: 400 held-out pairs, the benchmark.

**The base checkpoint:** 
| | |
|---|---|
| on Hugging Face | [`OVHaiLLM/Olala-v1.0-mini-7A1B`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B) |
| in the image | downloaded to `/opt/ckpt/ovhaillm-rl` when the container starts |
| size | 13.6 GB: bf16 weights, plus 0.07 GB of small fp32 tensors |

- **The image doesn't contain it:** when a container starts, it downloads the checkpoint from Hugging Face ([`OVHaiLLM/Olala-v1.0-mini-7A1B`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B), about 14 GB, a few minutes), then runs the command. If a checkpoint is already mounted at `/opt/ckpt/ovhaillm-rl`, nothing is downloaded. `CKPT_REPO` changes the repo, `CKPT_REV` picks a branch, tag or commit.
- **For Colab or local Python:** download the same files: `hf download OVHaiLLM/Olala-v1.0-mini-7A1B --local-dir olala-ckpt`.
- **Reasoning:** its chat template supports a reasoning mode. The tutorial turns it off (`reasoning_effort: disabled`), because the translation data has no reasoning traces: the model answers directly, with an empty analysis channel.

**Ways to run it:**

| | where | what you need | code |
|---|---|---|---|
| [A. OVH AI Training](#a-ovh-ai-training) | an OVH AI Training H100 job | data and code are in the image, the model is downloaded at start-up | `sft/` (custom LoRA trainer) |
| [A2. Locally with Docker](#a2-locally-with-docker) | your own H100 | Docker + NVIDIA Container Toolkit | the same image as A |
| [B. Google Colab](#b-google-colab) | a Colab A100 / H100 runtime | the checkpoint and the data are downloaded | `colab/olala_translation_sft.ipynb` (TRL) |
| [C. Local Python](#c-local-python) | your own A100 / H100 | the checkpoint on disk, and the Python env | `colab/translation_sft.py` (TRL, same as B) |

## A. OVH AI Training

One H100 job runs the full pipeline: benchmark the base → LoRA SFT → merge → benchmark the result → table. It uses the custom trainer in `sft/`, which gives more settings, a merged checkpoint, and per-sentence results in JSON. The image is public on Docker Hub, [`ovhaillm/olala-sft-tutorial`](https://hub.docker.com/r/ovhaillm/olala-sft-tutorial), so OVH pulls it with no setup (no `ovhai registry add`):

```
ovhaillm/olala-sft-tutorial:20261117
```

| | |
|---|---|
| base checkpoint | `/opt/ckpt/ovhaillm-rl`, downloaded when the job starts |
| data | `/opt/mt/data/open` |
| code | `/opt/mt/sft` |
| config | `/opt/mt/sft/configs/mt_lora_ecb.yaml`: LoRA r32, 2 epochs (500 steps), lr 5e-5 |

**1. Start a job:**

```bash
ovhai job run --name mt-sft --flavor h100-1-gpu --flavor-count 1 --timeout 8h \
    --volume ovhaillm-mt-sft-out@GRA/run1:/workspace/out:rw \
    ovhaillm/olala-sft-tutorial:20261117 \
    -- sleep infinity
```

- **The volume keeps the results:** without it, `/workspace/out` is lost when the job ends.
- **Always give the timeout a unit:** a bare `--timeout 28` means 28 seconds.
- **To copy the results over SSH later,** add `--ssh-public-keys ~/.ssh/id_ed25519.pub` now: it can't be added once the job runs (see "Getting the fine-tuned checkpoint to your machine" below).

When `ovhai job list` shows RUNNING, open a shell: `ovhai job exec -it <job-id> -- bash`. The checkpoint download runs before `sleep infinity`, so it's already in `/opt/ckpt/ovhaillm-rl` by then.

Or launch it directly with `ovh/launch_sft_baked.sh`, which runs `run_sft_bench.sh` as the job's command.

**2. Run it.** All in one (base benchmark → training → merge → fine-tuned benchmark → table), about 2 hours:

```bash
nohup run_sft_bench.sh > /workspace/out/run.log 2>&1 &
tail -f /workspace/out/run.log
```

For a quick check first, run `OUT=/workspace/out/check BENCH_N=20 run_sft_bench.sh training.max_steps=20`.

Or step by step:

```bash
cd /opt/mt/sft; OUT=/workspace/out
python bench_mt.py --model-dir /opt/ckpt/ovhaillm-rl --label base --out $OUT/bench_base.json      # benchmark the base
./train.sh configs/mt_lora_ecb.yaml output_dir=$OUT merge_when_done=true                        # LoRA SFT + merge
python bench_mt.py --model-dir $OUT/merged --label finetuned --out $OUT/bench_finetuned.json     # benchmark the result
python bench_mt.py --compare $OUT/bench_base.json $OUT/bench_finetuned.json                     # the table
```

- **Change any setting** with `key=value` arguments, e.g. `training.learning_rate=1e-4 lora.r=16 training.num_train_epochs=1`.
- **Benchmark size:** `bench_mt.py --max-n 50` uses 50 sentences instead of 400.

**3. Results,** in `/workspace/out` (and in the bucket once the job stops):
- `bench.txt`: the score table.
- `bench_*.json`: every translation.
- `adapter/`: the LoRA adapter.
- `merged/`: the full fine-tuned model (13.6 GB).

Stop the job with `ovhai job stop <job-id>`. It keeps billing until you do, or until its timeout.

### Getting the fine-tuned checkpoint to your machine

The simplest way is `rsync` over SSH while the job runs.

**1. Enable SSH when you launch** (it can't be added later): add `--ssh-public-keys ~/.ssh/id_ed25519.pub` to `ovhai job run`, or with the script:

```bash
SSH_KEY=~/.ssh/id_ed25519.pub KEEP_ALIVE=1 ovh/launch_sft_baked.sh
```

**2. Keep the job alive after training.** A job stops as soon as its command ends, and SSH goes with it. A job started with `-- sleep infinity` (step 1 above) stays up; with the script, `KEEP_ALIVE=1` does the same (the log is in `/workspace/out/run.log`).

**3. Copy, once training is done.** `ovhai job get <job-id>` shows the address under `Ssh Url`:

```bash
rsync -avP <ssh-url>:/workspace/out/merged  ./results/     # the full model
rsync -avP <ssh-url>:/workspace/out/adapter ./results/     # or just the LoRA adapter
```

`rsync -P` resumes an interrupted copy, which helps for 13.6 GB. `scp -r` works too, but restarts from zero.

**4. Stop the job:** `ovhai job stop <job-id>`.

**Other ways:**
- **No SSH:** stream a tar archive through `ovhai job exec`, without `-it`, which would corrupt the data: `ovhai job exec <job-id> -- tar czf - -C /workspace/out adapter | tar xzf - -C ./results/`.
- **The output bucket:** with a volume, `ovhai job push-data <job-id>` copies `/workspace/out` to the bucket while the job runs, and OVH copies it anyway when the job ends. Download with `ovhai bucket object download ovhaillm-mt-sft-out@GRA --prefix run1/ -o ./results/`.

These copy methods haven't been tried with this image yet. Check the copy with `sha256sum` on both sides.

**Using the result:** `merged/` loads like the base, with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`. For the adapter alone, load the base, then `PeftModel.from_pretrained(base_model, "results/adapter")`.

## A2. Locally with Docker

The same image runs on your own machine.

**You need:**
- **A GPU:** an NVIDIA H100. That's what it was tested on, and FlashAttention 3 is built for H100 (`sm_90`).
- **A driver** that supports CUDA 12.8 (R570 or newer).
- **Docker** with the NVIDIA Container Toolkit.
- **About 40 GB of disk:** 24.3 GB for the image and 13.6 GB for the downloaded checkpoint, plus the outputs.

**1. Pull** (once), from Docker Hub, [`ovhaillm/olala-sft-tutorial`](https://hub.docker.com/r/ovhaillm/olala-sft-tutorial); no login needed:

```bash
docker pull ovhaillm/olala-sft-tutorial:20261117
```

**2. Run.**

```bash
IMG=ovhaillm/olala-sft-tutorial:20261117
mkdir -p out

# the pipeline of A (about 2 h on a free H100): table in ./out/bench.txt, merged model in ./out/merged
docker run --rm --gpus '"device=0"' --ipc=host -v $PWD/out:/workspace/out $IMG run_sft_bench.sh

# quick check of the pipeline
docker run --rm --gpus '"device=0"' --ipc=host -v $PWD/out:/workspace/out -e BENCH_N=20 $IMG \
    run_sft_bench.sh training.max_steps=20

# a shell inside, to explore
docker run --rm -it --gpus '"device=0"' --ipc=host -v $PWD/out:/workspace/out $IMG bash
```

With no command after the image, it runs the TRL version instead, `colab/translation_sft.py` (the same code as the Colab notebook, B): table at the end, LoRA adapter in `./out/lora`.

Each `docker run` downloads the checkpoint again, because the container starts empty. To download it once, keep it in a folder and mount that folder:

```bash
hf download OVHaiLLM/Olala-v1.0-mini-7A1B --local-dir $PWD/olala-ckpt          # once
docker run --rm --gpus '"device=0"' --ipc=host -v $PWD/out:/workspace/out \
    -v $PWD/olala-ckpt:/opt/ckpt/ovhaillm-rl:ro $IMG run_sft_bench.sh            # no download
```

The step-by-step commands are in A.

**Things to know:**
- **`--ipc=host` is needed:** without it, the data loaders run out of shared memory.
- **Pick a free GPU:** `"device=0"` is GPU 0. On a GPU shared with another job, everything is several times slower.
- **Output files are owned by root** on standard Docker, because the image runs as root: `sudo chown -R $USER out` afterwards. With rootless Docker they're already yours.
- **Don't add `--user <your uid>`:** the image has an account only for root and OVH's uid 42420, and an unknown uid can crash `torch.compile` (`getpwuid(): uid not found`).
- **Internet is needed only for the checkpoint download** at start-up (skipped with a mounted checkpoint). Everything else runs offline.

## B. Google Colab

Open `colab/olala_translation_sft.ipynb` in Colab and pick an **A100 or H100** runtime (*Runtime → Change runtime type*). The model's kernels don't run on a T4.

**Nothing to upload.** The notebook downloads the starting checkpoint from [`OVHaiLLM/Olala-v1.0-mini-7A1B`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B) (13.6 GB), and the data (`ecb_enfr_train.jsonl`, `ecb_enfr_eval.jsonl`) from this repository's `open_data/` folder on GitHub. Both downloads need the repos to be public.

Then run the cells in order:
1. Install the pinned libraries (about 5 minutes).
2. Mount Google Drive (a sign-in window, for saving the adapter at the end), then download the checkpoint and the data to the runtime's disk (a few minutes, on every new runtime).
3. Load the model.
4. Evaluate the base.
5. LoRA SFT: 1 epoch, 250 steps, about 30–45 minutes on an A100.
6. Evaluate again and print the before/after table.
7. Save the LoRA adapter to Google Drive, `My Drive/olala-sft/lora` (change `DRIVE_DIR` in the loading cell for another folder).

The two evaluations decode 400 sentences each, one at a time, which is roughly 10–15 minutes each.

## C. Local Python

The same tutorial as B, as a script, on your own GPU.

**You need:**
- **A GPU:** an A100 or H100, with a driver that supports CUDA 12.8 (R570 or newer).
- **Linux x86_64 and Python 3.12,** the same as Colab.
- **Possibly the CUDA toolkit** (`nvcc`, 12.8). The Mamba-3 kernels (TileLang) compile at run time. Colab has the toolkit preinstalled, but a bare machine may not.
- **The base checkpoint folder on disk,** with the files listed in B.

**1. Install** (once). These are the same steps as the notebook's install cell, in a fresh venv:

```bash
python3.12 -m venv mt-venv && source mt-venv/bin/activate

# the libraries pinned in the notebook
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch==2.11.0 transformers==5.8.0 trl==1.5.1 peft==0.19.1 accelerate==1.14.0 datasets sacrebleu \
    tilelang==0.1.9 apache-tvm-ffi==0.1.9 torch-c-dlpack-ext==0.1.5 nvidia-cutlass-dsl==4.5.2 cuda-python==12.9.7 \
    quack-kernels==0.5.0 z3-solver==4.15.4.0 einops==0.8.2 flash-linear-attention==0.5.2 fla-core==0.5.2 ninja

# FlashAttention 2, prebuilt for torch 2.11 (without it, the model falls back to the slower flex attention)
pip install --no-deps https://github.com/erictang000/flash-attention/releases/download/v2.8.3-torch2.11-clean/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl

# the Mamba-3 and ScatterMoE kernels (pure Python, compiled on first use), at the notebook's commits
for spec in OVHai-LLM/mamba@039a089 shawntan/scattermoe@47b5e15; do
    repo=${spec%@*}; d=$(basename $repo)
    [ -d $d ] || git clone -q --filter=blob:none https://github.com/$repo $d
    git -C $d checkout -q ${spec#*@}
done
printf "$PWD/mamba\n$PWD/scattermoe\n" > "$(python -c 'import site; print(site.getsitepackages()[0])')/olala_kernels.pth"
bash scripts/apply_olala_training_fixes.sh --python "$(which python)" --scattermoe "$PWD/scattermoe"
```

The three GitHub repos: [`OVHai-LLM/mamba`](https://github.com/OVHai-LLM/mamba) (the Mamba-3 kernels), [ScatterMoE](https://github.com/shawntan/scattermoe) (the experts) and [`OVHai-LLM/olala-v1`](https://github.com/OVHai-LLM/olala-v1) (the model code and the training fixes).

**2. Run,** from the repo:

```bash
python colab/translation_sft.py --model-dir /path/to/checkpoint                                # the full run
python colab/translation_sft.py --model-dir /path/to/checkpoint --num-train 64 --max-eval 20   # quick check
```

- **Data:** it reads `open_data/ecb_enfr_train.jsonl` and `open_data/ecb_enfr_eval.jsonl` from the repo by default. Change them with `--train-file` and `--eval-file`.
- **Output:** the before / after table at the end, and the LoRA adapter in `mt-sft/lora` (change it with `--output-dir`).
- **Time:** about 1.5 hours on a free H100: two evaluation passes of 400 sentences, and 250 training steps.
- **Tested so far:** the script, in our Docker environment. This install on a bare machine hasn't been run yet; if a step fails, the Docker image (A2) has a working environment.

## Reference results

From version A (custom trainer: LoRA r32, 2 epochs, lr 5e-5), on the 400 held-out ECB sentences. All gains are significant (paired bootstrap, p = 0.001).

| | base BLEU | fine-tuned BLEU | base chrF++ | fine-tuned chrF++ |
|---|---|---|---|---|
| en→fr | 42.1 | **45.6** | 63.7 | **66.5** |
| fr→en | 38.3 | **44.2** | 63.8 | **67.1** |

Versions B and C use TRL with slightly different settings: LoRA r16, 1 epoch, lr 1e-4. Their scores should be close to these, but not identical. Their reference numbers will be added after the first full run on OVH; so far the TRL version has only been run as a pipeline check (32 pairs, 6 sentences).

