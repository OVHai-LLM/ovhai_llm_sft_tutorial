"""Fine-tune the model for English<->French translation with TRL (SFT + LoRA).

Plain-Python version of the Colab notebook in colab/. It trains on in-domain (central banking /
finance) sentence pairs from the open ECB corpus (open_data/ecb_enfr_train.jsonl, built by
open_data/make_ecb_enfr.py), evaluates BLEU / chrF++ on 400 held-out ECB pairs before and after
training, then saves the LoRA adapter.

Requirements: an A100/H100 GPU and the training environment (the notebook's install cell sets it up), i.e.
    torch==2.11.0 transformers==5.8.0 trl==1.5.1 peft==0.19.1 accelerate==1.14.0 datasets sacrebleu
    tilelang==0.1.9 apache-tvm-ffi==0.1.9 nvidia-cutlass-dsl==4.5.2 quack-kernels==0.5.0 flash-linear-attention==0.5.2
    flash-attn 2.8.3 (the torch 2.11 wheel the notebook installs)
    + OVHai-LLM/mamba@039a089 and shawntan/scattermoe@47b5e15 on PYTHONPATH, with the training fixes applied

Usage:
    python translation_sft.py --model-dir /path/to/checkpoint
    python translation_sft.py --model-dir ... --num-train 64 --max-eval 20      # quick check

--train-file: `messages` jsonl rows (open_data/ format). --eval-file: `messages` rows, or rows with
source_language / target_language / source / reference / domain.
"""
import argparse
import json
import os
from pathlib import Path

import sacrebleu
import torch
from datasets import Dataset
from peft import LoraConfig
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", default=os.environ.get("MODEL_DIR"), required="MODEL_DIR" not in os.environ,
                    help="base checkpoint directory (default: $MODEL_DIR, set in the Docker image)")
# The ECB files: open_data/ next to colab/ in the repo, or $OPEN_DATA_DIR (set in the Docker image).
OPEN_DATA = Path(os.environ.get("OPEN_DATA_DIR") or Path(__file__).resolve().parent.parent / "open_data")
parser.add_argument("--train-file", default=str(OPEN_DATA / "ecb_enfr_train.jsonl"))
parser.add_argument("--eval-file", default=str(OPEN_DATA / "ecb_enfr_eval.jsonl"))
parser.add_argument("--output-dir", default="mt-sft")
parser.add_argument("--num-train", type=int, default=None, help="use the first N training rows (default: all)")
parser.add_argument("--max-eval", type=int, default=None, help="use N eval sentences, half per direction (default: all)")
# Upstream mamba_ssm (the Docker image) reads its saved tensors twice in backward, which non-reentrant
# checkpointing rejects ("Unpack is being triggered for a tensor that was already unpacked once").
# The OVHai-LLM/mamba fork used on Colab does not, so the default stays as it was there.
parser.add_argument("--use-reentrant", action="store_true", default=os.environ.get("USE_REENTRANT") == "1",
                    help="reentrant gradient checkpointing (default on in the Docker image)")
args = parser.parse_args()

# ---------------------------------------------------------------- model
tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
tokenizer.padding_side = "right"  # the model relies on causality to ignore padding: pad at the end
model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")
print(f"Model loaded: {model.num_parameters():,} parameters")

# ---------------------------------------------------------------- prompts
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


# ---------------------------------------------------------------- data
def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def to_eval_row(r):
    """A `messages` row -> source_language / target_language / source / reference / domain."""
    if "messages" not in r:
        return r
    request = json.loads(next(m["content"] for m in r["messages"] if m["role"] == "user"))
    return {"source_language": request["source_language"], "target_language": request["target_language"],
            "source": request["source_text"], "domain": request.get("domain", "General"),
            "reference": next(m["content"] for m in r["messages"] if m["role"] == "assistant")}


# Training rows are already in the chat format, with chat_template_kwargs {"reasoning_effort": "disabled"}.
train_dataset = Dataset.from_list(read_jsonl(args.train_file)[:args.num_train])
eval_rows = [to_eval_row(r) for r in read_jsonl(args.eval_file)]
if args.max_eval:  # keep both directions balanced
    eval_rows = [r for src in ("English", "French") for r in [x for x in eval_rows if x["source_language"] == src][:args.max_eval // 2]]
eval_dataset = Dataset.from_list(eval_rows)
print(train_dataset, eval_dataset, sep="\n")


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def translate(model, text, src, tgt, **kwargs):
    inputs = tokenizer.apply_chat_template(build_messages(text, src, tgt, **kwargs), add_generation_prompt=True,
                                           reasoning_effort="disabled", return_tensors="pt", return_dict=True).to("cuda")
    output = model.generate(**inputs, max_new_tokens=384, do_sample=False, use_cache=True)
    answer = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:])  # ...final<|content|>TRANSLATION<|channel_end|>
    return answer.split("<|content|>")[-1].split("<|channel_end|>")[0].strip()


def evaluate(model):
    model.eval()
    hyps = [translate(model, r["source"], r["source_language"], r["target_language"], domain=r["domain"])
            for r in tqdm(eval_dataset)]
    scores = {}
    for src, tgt in [("English", "French"), ("French", "English")]:
        idx = [i for i, r in enumerate(eval_dataset) if r["source_language"] == src]
        h, refs = [hyps[i] for i in idx], [[eval_dataset[i]["reference"] for i in idx]]
        scores[f"{src[:2].lower()}->{tgt[:2].lower()}"] = (sacrebleu.corpus_bleu(h, refs).score,
                                                           sacrebleu.corpus_chrf(h, refs, word_order=2).score)
    return hyps, scores


baseline_hyps, baseline = evaluate(model)
for pair, (bleu, chrf) in baseline.items():
    print(f"baseline {pair}: BLEU {bleu:.1f} | chrF++ {chrf:.1f}")

# ---------------------------------------------------------------- training
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["in_proj", "c_q", "mixer_proj", "fc_1", "fc_2", "down_proj", "up_proj"],
    task_type="CAUSAL_LM",
)
sft_config = SFTConfig(
    output_dir=args.output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch size of 16 sentence pairs
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    max_length=768,
    assistant_only_loss=True,  # loss on the translation only, not the prompt
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": args.use_reentrant},
    bf16=True,
    logging_steps=5,
    save_strategy="no",
    report_to="none",
)
trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_dataset,
                     processing_class=tokenizer, peft_config=peft_config)
trainer.model.print_trainable_parameters()
trainer.train()
print(f"Peak GPU memory: {torch.cuda.max_memory_reserved() / 1e9:.1f} GB")

# ---------------------------------------------------------------- results
finetuned_hyps, finetuned = evaluate(trainer.model)
print(f"{'':<8}{'BLEU before':>13}{'BLEU after':>12}{'chrF++ before':>15}{'chrF++ after':>14}")
for pair in baseline:
    print(f"{pair:<8}{baseline[pair][0]:>13.1f}{finetuned[pair][0]:>12.1f}{baseline[pair][1]:>15.1f}{finetuned[pair][1]:>14.1f}")

for i in sorted({0, 1, len(eval_dataset) // 2, len(eval_dataset) // 2 + 1}):
    r = eval_dataset[i]
    print(f"\n{r['source_language']} -> {r['target_language']} ({r['domain']})\n   {r['source']}")
    print(f"   reference:  {r['reference']}\n   original:   {baseline_hyps[i]}\n   fine-tuned: {finetuned_hyps[i]}")

trainer.save_model(f"{args.output_dir}/lora")
print(f"\nLoRA adapter saved to {args.output_dir}/lora")
# Merged standalone checkpoint:
# merged = trainer.model.merge_and_unload(); merged.save_pretrained(f"{args.output_dir}/merged"); tokenizer.save_pretrained(f"{args.output_dir}/merged")
