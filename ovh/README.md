# Running on OVHcloud AI Training

These scripts wrap `ovhai job run`. **They haven't been run on OVH yet.** The images they launch were tested locally the way OVH runs them: as uid 42420, with no Hugging Face hub access.

| script | image | what it runs |
|---|---|---|
| `launch_sft_baked.sh` | `ovhaillm/olala-sft-tutorial:20261117` (public, Docker Hub) | `run_sft_bench.sh`: benchmark → in-domain (finance) LoRA SFT with the custom trainer → benchmark, with open ECB data; the checkpoint is downloaded at start-up |
| `launch_sft.sh` | `olala-mt-trainer:20261107-0703` | `sft/sft_lora.py` — checkpoint and data come from Object Storage |

## One-time setup

1. **Install the CLI and log in:**
   ```bash
   curl -sSL https://cli.gra.ai.cloud.ovh.net/install.sh | bash
   ovhai login
   ```
2. **Only for `launch_sft.sh`: let OVH pull from the corp registry.** The tutorial image is public on Docker Hub and needs nothing. The trainer image is in the GitLab project's registry: use a deploy token from the GitLab project `Jingshu/olala-mt-tuto` (*Settings → Repository → Deploy tokens*) with the `read_registry` scope:
   ```bash
   ovhai registry add git.corp.linguacustodia.com:5050 --username <deploy-token-user> --password <deploy-token>
   ```
   Check that OVH can reach the registry. If `git.corp.linguacustodia.com:5050` isn't reachable from the internet, the job will hang at image pull.
3. **Find the H100 flavor name:**
   ```bash
   ovhai capabilities flavor list
   ```
   The scripts default to `h100-1-gpu`; override it with `FLAVOR=...` if the name differs.

## LoRA SFT with a benchmark (the tutorial)

The tutorial image (`ovhaillm/olala-sft-tutorial:20261117`, public on Docker Hub) holds small open datasets, a benchmark and the custom trainer, `sft/`, and downloads the base checkpoint from Hugging Face (`OVHaiLLM/Olala-v1.0-mini-7A1B`) when the job starts. The only volume is the output bucket. No in-house data is in the image.

- **Base checkpoint:** [`OVHaiLLM/Olala-v1.0-mini-7A1B`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B), downloaded into `/opt/ckpt/ovhaillm-rl` when the job starts. It's the same checkpoint the Colab notebook uses.
- **Data (in-domain, the default):** central banking and finance en↔fr from the OPUS [ECB corpus](https://opus.nlpl.eu/ECB-v1.php): European Central Bank publications, professionally translated. `open_data/make_ecb_enfr.py` builds it:
  - detokenizes it, and drops PDF damage (lost accents, stray diacritics), boilerplate, tables and misaligned pairs (LaBSE similarity < 0.80);
  - keeps 4000 pairs for training and 400 for eval and benchmark, half in each direction;
  - holds the eval pairs out by page, with no 8-word overlap with the training set, so the benchmark measures domain adaptation and not memorized ECB formulas.

  The prompt is the tutorial's, with `"domain": "Finance"` and `reasoning_effort: disabled`, so the model learns to answer directly with an empty analysis channel.
- **General-domain check:** 2000 + 200 opus-100 pairs are in the image too. Benchmark on them to see whether the finance SFT hurts generic translation: `python bench_mt.py --model-dir ... --eval-file /opt/mt/data/open/opus100_enfr_eval.jsonl`. You can also train on them instead with `sft/configs/mt_lora_baked.yaml`.
- **What a job does** (`sft/run_sft_bench.sh`):
  1. Benchmarks the base: BLEU and chrF++ per direction, greedy, reasoning disabled.
  2. Trains LoRA.
  3. Merges the adapter into the base.
  4. Benchmarks the merged model.
  5. Prints both scores side by side.

```bash
ovh/launch_sft_baked.sh                                    # sft/configs/mt_lora_ecb.yaml as is
BENCH_N=50 ovh/launch_sft_baked.sh training.max_steps=20   # quick check: 50 benchmark sentences, 20 steps
ovh/launch_sft_baked.sh lora.r=16 training.num_train_epochs=1
```

| in the image | |
|---|---|
| `/opt/ckpt/ovhaillm-rl` | the base checkpoint (`model_path`), downloaded when the job starts |
| `/opt/mt/data/open` | `ecb_enfr_{train,eval}.jsonl` (the default), `opus100_enfr_{train,eval}.jsonl` (general domain), and the scripts that made them |
| `/workspace/out` | output, synced to bucket `ovhaillm-mt-sft-out` under a prefix named after the job |

**Copying the outputs to your machine:** launch with `SSH_KEY=~/.ssh/id_ed25519.pub KEEP_ALIVE=1`, then `rsync -avP <ssh-url>:/workspace/out/merged ./results/`. Details in [`TUTORIAL_SFT.md`](../TUTORIAL_SFT.md), section A.

**Output:**
- `bench.txt`: the score table, plus two example translations.
- `bench_base.json` and `bench_finetuned.json`: scores and every translation.
- `adapter/` and `merged/`.
- `checkpoints/` and `sft_config.yaml`.

**Time:** the benchmark decodes one sentence at a time, about 15 minutes per 200 sentences on an H100, and it runs twice (400 sentences each). Training is 500 steps: 2 epochs of 4000 rows, 16 rows per step, at lr 5e-5.

**Changing the checkpoint or data** means a new image: rebuild as in the repository README's *Rebuilding* (`CKPT` is required; `open_data/make_ecb_enfr.py` and `make_opus100_enfr.py` regenerate the data), then update the tag in `launch_sft_baked.sh`. Only the changed layers are pushed.

## Generic LoRA SFT (checkpoint and data from Object Storage)

Upload the base checkpoint and the data to Object Storage first, for example:

```bash
ovhai bucket object upload olala-ckpt@GRA /path/to/checkpoint/*    # files at the bucket root
ovhai bucket object upload olala-mt-data@GRA train.jsonl
```

Then launch:

```bash
CKPT_BUCKET=olala-ckpt DATA_BUCKET=olala-mt-data TRAIN_FILES='[/workspace/data/train.jsonl]' \
    ovh/launch_sft.sh completion_field=chosen
```

- **Overrides:** any `key=value` after the script name is passed to `sft_lora.py`; see the repository README for the keys.
- **More GPUs:** `GPU=2 FLAVOR=...` runs DDP over them. Multi-GPU hasn't been tested yet.

## Watching a job

```bash
ovhai job list
ovhai job logs -f <job-id>
ovhai job get <job-id>
ovhai job stop <job-id>
ovhai bucket object list ovhaillm-mt-sft-out@GRA           # outputs after the job ends
```

Check the logs for `Using attention implementation: fa3`. If they say `fa2`, the wrong image is running.
