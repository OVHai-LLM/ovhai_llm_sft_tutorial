"""Build olala_translation_sft.ipynb:  python build_notebook.py olala_translation_sft.ipynb"""
import json
import sys

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(keepends=True)})


md(r"""
# Fine-Tuning Olala for Translation with TRL (SFT)

This notebook fine-tunes **Olala 7A1B** (hybrid Mamba-3 / attention, MoE with 256 experts, 6 active) for
English↔French translation with Supervised Fine-Tuning (SFT), [TRL](https://github.com/huggingface/trl) and LoRA.
It trains on central banking / finance sentence pairs from the open ECB corpus and measures BLEU and chrF++
before and after training, on 400 held-out ECB sentences.

## 🎯 What you'll find

- **The task**: a JSON translation request goes in (languages, text, terminology, domain), the translation comes out
- **LoRA + SFT**: train on the translations only, with reasoning disabled
- **Evaluation**: BLEU and chrF++ per direction, before and after training, and side-by-side examples

## ⚙️ Before you start

- **GPU**: an **A100 or H100** runtime (*Runtime → Change runtime type*). Olala's kernels do not run on the T4.
- **The starting checkpoint** is downloaded from Hugging Face,
  [`OVHaiLLM/Olala-v1.0-mini-7A1B`](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B) (13.6 GB, a few minutes).
- **The data** is downloaded from the tutorial's GitHub repository
  ([`OVHai-LLM/ovhai_llm_sft_tutorial`](https://github.com/OVHai-LLM/ovhai_llm_sft_tutorial), `open_data/`).
- **Nothing to upload.** At the end, the LoRA adapter is saved to Google Drive (`My Drive/olala-sft/lora`).
""")

md(r"""
# 📦 Installation & Setup

We pin every library to the versions Olala was validated with. Olala's Mamba-3 kernels come from the
[`OVHai-LLM/mamba`](https://github.com/OVHai-LLM/mamba) fork, and its experts from [ScatterMoE](https://github.com/shawntan/scattermoe).
The training fixes (a few patches to ScatterMoE) come from this tutorial's repository, `script/apply_olala_training_fixes.sh`.
Both are pure Python with kernels compiled on first use, so they need no build step. Attention uses
[FlashAttention 2](https://github.com/Dao-AILab/flash-attention), from a prebuilt wheel for torch 2.11 (the one Olala's
environment pins); without it, the model falls back to PyTorch's slower flex attention. This takes about 5 minutes.
""")

code(r'''
import os, site, subprocess, sys

def sh(cmd): subprocess.run(cmd, shell=True, check=True)

sh("pip uninstall -q -y torchvision torchaudio")
sh("pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128")
sh("pip install -q torch==2.11.0 transformers==5.8.0 trl==1.5.1 peft==0.19.1 accelerate==1.14.0 datasets sacrebleu "
   "tilelang==0.1.9 apache-tvm-ffi==0.1.9 torch-c-dlpack-ext==0.1.5 nvidia-cutlass-dsl==4.5.2 cuda-python==12.9.7 "
   "quack-kernels==0.5.0 z3-solver==4.15.4.0 einops==0.8.2 flash-linear-attention==0.5.2 fla-core==0.5.2 ninja")
sh("pip install -q --no-deps https://github.com/erictang000/flash-attention/releases/download/v2.8.3-torch2.11-clean/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl")
for repo, commit in [("OVHai-LLM/mamba", "039a089"), ("shawntan/scattermoe", "47b5e15")]:
    d = "/content/" + repo.split("/")[1]
    sh(f"[ -d {d} ] || git clone -q --filter=blob:none https://github.com/{repo} {d}; git -C {d} checkout -q {commit}")
open(f"{site.getsitepackages()[0]}/olala_kernels.pth", "w").write("/content/mamba\n/content/scattermoe\n")
sh("curl -sSfL -o /content/apply_olala_training_fixes.sh https://raw.githubusercontent.com/OVHai-LLM/ovhai_llm_sft_tutorial/main/script/apply_olala_training_fixes.sh")
sh(f"bash /content/apply_olala_training_fixes.sh --python {sys.executable} --scattermoe /content/scattermoe")
''')

md(r"""
# Loading the model from Hugging Face 🤗

We mount Google Drive first, where the LoRA adapter is saved at the end (a sign-in window asks for access), so the run
doesn't wait for it after training. Then we download the checkpoint and the data to the runtime's local disk (a new
runtime starts empty, so this runs each time). Olala ships its modeling code with the checkpoint, so we load
it with `trust_remote_code=True`.
""")

code(r'''
import json
import torch
import urllib.request
from google.colab import drive
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

drive.mount("/content/drive")
DRIVE_DIR = "/content/drive/MyDrive/olala-sft"   # where the LoRA adapter is saved; created if missing

CKPT_REPO = "OVHaiLLM/Olala-v1.0-mini-7A1B"
MODEL_DIR = "/content/olala-ckpt"
print(f"⬇️ Downloading {CKPT_REPO}...")
snapshot_download(CKPT_REPO, local_dir=MODEL_DIR)

DATA_URL = "https://raw.githubusercontent.com/OVHai-LLM/ovhai_llm_sft_tutorial/main/open_data"
DATA_DIR = "/content/olala-data"
os.makedirs(DATA_DIR, exist_ok=True)
for name in ["ecb_enfr_train.jsonl", "ecb_enfr_eval.jsonl"]:
    urllib.request.urlretrieve(f"{DATA_URL}/{name}", f"{DATA_DIR}/{name}")

print("📚 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.padding_side = "right"   # Olala relies on causality to ignore padding: pad at the end

print("🧠 Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
print(f"✅ Model loaded: {model.num_parameters():,} parameters")
''')

md(r"""
# 📚 Build the prompts and the datasets

Olala-MT is prompted with a **system prompt** and a **JSON request**, and answers with the translation only.
We use the non-reasoning mode (`reasoning_effort="disabled"`): the model answers right away. TRL reads the
`chat_template_kwargs` column and passes it to the chat template.
""")

code(r'''
SYSTEM_PROMPT = (
    "You are a professional translation assistant. Your task is to translate text carefully, pay attention to the "
    "following:\n1. Preserve the casing of words.\n2. Maintain the inline tags as-is in the translated text.\n"
    "3. Follow the terminology mappings provided by the user.\n4. The domain of the text.\n"
    "Return only the translation, without any additional explanation."
)

def build_messages(text, src, tgt, domain="General", terms=None, translation=None):
    request = {"source_language": src, "target_language": tgt, "source_text": text,
               "terminology_mapping": [{"source_term": s, "target_term": t} for s, t in (terms or {}).items()],
               "domain": domain}
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=2)}]
    if translation:
        messages.append({"role": "assistant", "content": translation})
    return messages

print(build_messages("Le chiffre d'affaires a augmenté de 5 %.", "French", "English",
                     terms={"chiffre d'affaires": "revenue"})[1]["content"])
''')

md(r"""
**Training data:** `ecb_enfr_train.jsonl`, 4,000 sentence pairs from the [ECB corpus](https://opus.nlpl.eu/ECB-v1.php)
(European Central Bank publications: monetary policy, banking supervision, payments, statistics), alternating
English→French and French→English, with `"domain": "Finance"`. The rows are already in the chat format above.
`open_data/make_ecb_enfr.py` in the repo builds them: it detokenizes the corpus and drops PDF damage, boilerplate and
misaligned pairs.

**Evaluation data:** `ecb_enfr_eval.jsonl`, 200 English→French and 200 French→English pairs, held out by page and
sharing no 8-word run with the training set, so the score measures domain adaptation, not memorized ECB formulas.
""")

code(r'''
from datasets import Dataset

def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]

def to_eval_row(r):   # a `messages` row -> source / reference and their languages and domain
    request = json.loads(r["messages"][1]["content"])
    return {"source_language": request["source_language"], "target_language": request["target_language"],
            "source": request["source_text"], "domain": request["domain"], "reference": r["messages"][2]["content"]}

train_dataset = Dataset.from_list(read_jsonl(f"{DATA_DIR}/ecb_enfr_train.jsonl"))
eval_dataset = Dataset.from_list([to_eval_row(r) for r in read_jsonl(f"{DATA_DIR}/ecb_enfr_eval.jsonl")])
print(train_dataset, eval_dataset, sep="\n")
print(train_dataset[0]["messages"][1]["content"])
''')

md(r"""
# 📏 Evaluate the model before training

We translate each evaluation sentence with greedy decoding, one at a time (Olala's cached generation does not
handle padded batches), keep the `final` channel of the answer, and score with [sacreBLEU](https://github.com/mjpost/sacrebleu).
""")

code(r'''
import sacrebleu
from tqdm.auto import tqdm

@torch.no_grad()
def translate(model, text, src, tgt, **kwargs):
    inputs = tokenizer.apply_chat_template(build_messages(text, src, tgt, **kwargs), add_generation_prompt=True,
                                           reasoning_effort="disabled", return_tensors="pt", return_dict=True).to("cuda")
    output = model.generate(**inputs, max_new_tokens=384, do_sample=False, use_cache=True)
    answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:])   # ...final<|content|>TRANSLATION<|channel_end|>
    return answer.split("<|content|>")[-1].split("<|channel_end|>")[0].strip()

def evaluate(model):
    model.eval()
    hyps = [translate(model, r["source"], r["source_language"], r["target_language"], domain=r["domain"])
            for r in tqdm(eval_dataset)]
    scores = {}
    for src, tgt in [("English", "French"), ("French", "English")]:
        idx = [i for i, r in enumerate(eval_dataset) if r["source_language"] == src]
        h, refs = [hyps[i] for i in idx], [[eval_dataset[i]["reference"] for i in idx]]
        scores[f"{src[:2].lower()}→{tgt[:2].lower()}"] = (sacrebleu.corpus_bleu(h, refs).score,
                                                          sacrebleu.corpus_chrf(h, refs, word_order=2).score)
    return hyps, scores

baseline_hyps, baseline = evaluate(model)
for pair, (bleu, chrf) in baseline.items():
    print(f"{pair}: BLEU {bleu:.1f} | chrF++ {chrf:.1f}")
''')

md(r"""
# 🎯 Supervised Fine-Tuning (SFT + LoRA)

## Wrap the model with PEFT

We adapt the large projections: the Mamba-3 input projection (`in_proj`), the attention queries (`c_q`), the mixers'
output projection (`mixer_proj`), and the MoE's shared expert (`fc_1`, `fc_2`) and expert projections (`down_proj`, `up_proj`).
The MoE router stays frozen. The 256 routed experts are fused ScatterMoE tensors, not linear layers, so LoRA does not apply to them.
""")

code(r'''
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["in_proj", "c_q", "mixer_proj", "fc_1", "fc_2", "down_proj", "up_proj"],
    task_type="CAUSAL_LM",
)
''')

md(r"""
## Launch training

`assistant_only_loss=True` computes the loss on the assistant turn only: the model learns to write the translation,
not to reproduce the prompt. One epoch over the 4,000 pairs is 250 steps. The first steps are slow while the kernels
compile; the run takes about 30–45 minutes on an A100.
""")

code(r'''
from trl import SFTConfig, SFTTrainer

sft_config = SFTConfig(
    output_dir="/content/olala-mt-sft",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # <- effective batch size of 16 sentence pairs
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    max_length=768,
    assistant_only_loss=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)
trainer.model.print_trainable_parameters()

print("🚀 Starting SFT training...")
trainer.train()
print("🎉 SFT training completed!")
print(f"🧮 Peak GPU memory: {torch.cuda.max_memory_reserved() / 1e9:.1f} GB")
''')

md("# 📊 Evaluate the fine-tuned model")

code(r'''
finetuned_hyps, finetuned = evaluate(trainer.model)

print(f"{'':<8}{'BLEU before':>13}{'BLEU after':>12}{'chrF++ before':>15}{'chrF++ after':>14}")
for pair in baseline:
    print(f"{pair:<8}{baseline[pair][0]:>13.1f}{finetuned[pair][0]:>12.1f}{baseline[pair][1]:>15.1f}{finetuned[pair][1]:>14.1f}")
''')

md("# 🔍 Original vs fine-tuned model")

code(r'''
for i in [0, 1, 100, 101]:
    r = eval_dataset[i]
    print(f"📄 {r['source_language']} → {r['target_language']} ({r['domain']})\n   {r['source']}")
    print(f"   reference:  {r['reference']}\n   original:   {baseline_hyps[i]}\n   fine-tuned: {finetuned_hyps[i]}\n")
''')

md(r"""
# 💾 Save and merge the model

We save the LoRA adapter to Google Drive (mounted at the start). Merging it into the base weights gives a standalone checkpoint (13.6 GB) that loads
like the original, with `trust_remote_code=True`.
""")

code(r'''
trainer.save_model(f"{DRIVE_DIR}/lora")
print(f"💾 LoRA adapter saved to: {DRIVE_DIR}/lora")

# To also save a merged checkpoint, uncomment:
# merged_model = trainer.model.merge_and_unload()
# merged_model.save_pretrained("/content/olala-mt-merged")
# tokenizer.save_pretrained("/content/olala-mt-merged")
''')

md(r"""
# 🚀 Next steps

- **Your own data**: replace the ECB pairs with sentence pairs from your domain, built with `build_messages` (fill
  `domain` and `terms` for terminology). Clean, in-domain references are what make fine-tuning pay off: on general
  data such as OPUS-100 subtitles, scores barely move, because the base model is already strong there.
- **Serving**: the merged model runs with Olala's vLLM fork, [`OVHai-LLM/vllm`](https://github.com/OVHai-LLM/vllm) (see the model card; use `--mamba-ssm-cache-dtype float32`).
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {sys.argv[1]}: {len(cells)} cells")
