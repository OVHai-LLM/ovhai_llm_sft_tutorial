#!/usr/bin/env python
"""LoRA SFT for OvhaiLLM checkpoints (hybrid Mamba-3 / attention, MoE).

    torchrun --standalone --nproc_per_node=N sft/sft_lora.py --config sft/configs/mt_lora.yaml [key=value ...]

Overrides use dotted keys: `lora.r=16 training.learning_rate=5e-5 train_files=[a.jsonl,b.jsonl]`.

DATA (jsonl / json / parquet). Each row is one of two shapes:

  * `prompt` + `completion` strings, already rendered with the model's chat template
    (the shape of rl_data/ and dpo_datasets/). Loss on the completion only.
    `completion_field` picks the column (`chosen` from a preference file, `ref`, ...);
    `wrap_completion: true` wraps raw text as
    `<|channel_start|>final<|content|>TEXT<|channel_end|><|im_end|>`.
  * `messages`: a chat, rendered with the checkpoint's own chat template. Loss on
    the assistant turns, from the template's {% generation %} markers. An optional
    `chat_template_kwargs` dict per row is passed to the template, e.g.
    {"reasoning_effort": "disabled"}.

Tokenization and masking happen here, not in TRL: what is trained is exactly the
prompt tokens + completion tokens, with no BOS/EOS added behind your back.

LORA TARGETS. The model has no q/k/v_proj. Every projection is the model's own
nn.Linear subclass: in_proj / in_proj_dyn (Mamba-3), c_q / W_A_* / W_B_*
(attention), gate_proj / mixer_proj (block), fc_1 / fc_2 (shared expert),
up_proj / down_proj (routed-expert in/out projections). The routed experts
themselves are 3-D tensors, not Linear, and the MoE router (moe_gate) is left
alone. Targets are resolved to full module names of nn.Linear instances, so a
layer type that does not have one of them is skipped instead of failing.

OUTPUT. `output_dir/adapter` holds the adapter; with `merge_when_done` (default),
`output_dir/merged` is a full checkpoint built by sft/merge_lora.py, which edits
the base safetensors in place of the targeted weights only -- every other tensor
is byte-identical to the base, fp32 ones included.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("sft_lora")

DEFAULT_TARGETS = [
    "in_proj", "in_proj_dyn",                     # Mamba-3 mixers
    "c_q", "W_A_k", "W_A_v", "W_B_k", "W_B_v",    # attention mixers
    "gate_proj", "mixer_proj",                    # block output gate + projection
    "fc_1", "fc_2",                               # shared expert
    "up_proj", "down_proj",                       # routed-expert in/out projections
]

FINAL_OPEN = "<|channel_start|>final<|content|>"
FINAL_CLOSE = "<|channel_end|><|im_end|>"

DEFAULTS: dict[str, Any] = {
    "model_path": None,
    "output_dir": None,
    "train_files": [],
    "eval_files": [],
    "prompt_field": "prompt",
    "completion_field": "completion",
    "messages_field": "messages",
    "wrap_completion": False,
    "max_length": 4096,
    # "drop" rows longer than max_length, or "truncate" them (keeps the prompt,
    # cuts the completion -- rarely what you want for MT).
    "overlong": "drop",
    "max_train_samples": None,
    "max_eval_samples": None,
    "seed": 42,
    "dataset_num_proc": 8,
    "merge_when_done": True,
    "lora": {
        "r": 32,
        "alpha": 64,
        "dropout": 0.05,
        "use_rslora": False,
        "target_leaf_names": DEFAULT_TARGETS,
        # Substrings of full module names to leave out, e.g. ["layers.0.", "mlp.up_proj"].
        "exclude": [],
    },
    # Passed to transformers.TrainingArguments as is; anything it accepts works.
    "training": {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 1.0e-4,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 50,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 3,
        "bf16": True,
        "gradient_checkpointing": True,
        # REENTRANT, against peft's usual advice. The mamba3 mimo kernel reads
        # ctx.saved_tensors twice in backward, which non-reentrant checkpointing
        # rejects: "Unpack is being triggered for a tensor that was already
        # unpacked once" (same finding in our RLHF runs).
        "gradient_checkpointing_kwargs": {"use_reentrant": True},
        # Reentrant checkpointing runs a second backward through each block,
        # which DDP only accepts with a static graph.
        "ddp_static_graph": True,
        "ddp_find_unused_parameters": False,
        "dataloader_num_workers": 2,
        "report_to": ["tensorboard"],
    },
}


# ----------------------------------------------------------------------------- config

def deep_update(base: dict, new: dict) -> dict:
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def apply_override(cfg: dict, item: str) -> None:
    if "=" not in item:
        raise SystemExit(f"override must be key=value, got {item!r}")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw)
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def load_config(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = ap.parse_args(argv)
    cfg = copy.deepcopy(DEFAULTS)
    with open(args.config) as f:
        deep_update(cfg, yaml.safe_load(f) or {})
    for o in args.overrides:
        apply_override(cfg, o)
    for key in ("model_path", "output_dir"):
        if not cfg[key]:
            raise SystemExit(f"{key} is required")
    if not cfg["train_files"]:
        raise SystemExit("train_files is required")
    if isinstance(cfg["train_files"], str):
        cfg["train_files"] = [cfg["train_files"]]
    if isinstance(cfg["eval_files"], str):
        cfg["eval_files"] = [cfg["eval_files"]]
    if cfg["overlong"] not in ("drop", "truncate"):
        raise SystemExit("overlong must be 'drop' or 'truncate'")
    return cfg


def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


# ----------------------------------------------------------------------------- data

def expand(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        if not hits:
            raise FileNotFoundError(f"no file matches {p}")
        paths.extend(hits)
    return paths


def load_rows(patterns: list[str]):
    import datasets

    parts = []
    for path in expand(patterns):
        kind = "parquet" if path.endswith(".parquet") else "json"
        parts.append(datasets.load_dataset(kind, data_files=path, split="train"))
    return parts[0] if len(parts) == 1 else datasets.concatenate_datasets(parts)


def tokenize_dataset(ds, tokenizer, cfg: dict, split: str):
    """-> columns input_ids, labels (with -100 on the prompt), length."""
    pf, cf, mf = cfg["prompt_field"], cfg["completion_field"], cfg["messages_field"]
    max_len, wrap, overlong = cfg["max_length"], cfg["wrap_completion"], cfg["overlong"]
    cols = set(ds.column_names)
    if mf in cols:
        mode = "messages"
    elif pf in cols and cf in cols:
        mode = "prompt_completion"
    else:
        raise SystemExit(
            f"{split}: rows need `{mf}`, or `{pf}` + `{cf}`; columns are {sorted(cols)}")
    eos = tokenizer.eos_token

    def one(row):
        if mode == "prompt_completion":
            completion = row[cf]
            if isinstance(completion, list):  # e.g. refs as a list: take the first
                completion = completion[0]
            if wrap:
                completion = f"{FINAL_OPEN}{completion}{FINAL_CLOSE}"
            p = tokenizer(row[pf], add_special_tokens=False)["input_ids"]
            c = tokenizer(completion, add_special_tokens=False)["input_ids"]
            ids, labels = p + c, [-100] * len(p) + c
            ends_eos = completion.endswith(eos)
        else:
            # Optional per-row template variables, e.g. {"reasoning_effort": "disabled"}
            # (same column name TRL uses). Rows without it render with the template defaults.
            out = tokenizer.apply_chat_template(
                row[mf], tokenize=True, return_dict=True, return_assistant_tokens_mask=True,
                **(row.get("chat_template_kwargs") or {}))
            ids = out["input_ids"]
            mask = out["assistant_masks"]
            labels = [t if m else -100 for t, m in zip(ids, mask)]
            ends_eos = True
        too_long = len(ids) > max_len
        if too_long and overlong == "truncate":
            ids, labels = ids[:max_len], labels[:max_len]
        return {"input_ids": ids, "labels": labels, "length": len(ids),
                "_drop": too_long and overlong == "drop",
                "_no_target": all(l == -100 for l in labels),
                "_no_eos": not ends_eos}

    out = ds.map(one, remove_columns=ds.column_names, num_proc=cfg["dataset_num_proc"],
                 desc=f"tokenize {split}")
    n = len(out)
    dropped, no_target, no_eos = sum(out["_drop"]), sum(out["_no_target"]), sum(out["_no_eos"])
    if is_main():
        logger.info("%s: %d rows (%s), %d over max_length=%d (%s), %d with no target tokens",
                    split, n, mode, dropped, max_len, overlong, no_target)
        if no_eos:
            logger.warning(
                "%s: %d/%d completions do not end with %r -- the model will not learn to stop. "
                "Set wrap_completion: true if the completion is raw text.", split, no_eos, n, eos)
    out = out.filter(lambda r: not r["_drop"] and not r["_no_target"])
    return out.remove_columns(["_drop", "_no_target", "_no_eos"])


class Collator:
    """Right-pad input_ids / labels to the longest row in the batch."""

    def __init__(self, pad_id: int, multiple_of: int = 8):
        self.pad_id, self.multiple_of = pad_id, multiple_of

    def __call__(self, rows):
        import torch

        n = max(len(r["input_ids"]) for r in rows)
        n = -(-n // self.multiple_of) * self.multiple_of
        ids = torch.full((len(rows), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(rows), n), -100, dtype=torch.long)
        attn = torch.zeros((len(rows), n), dtype=torch.long)
        for i, r in enumerate(rows):
            k = len(r["input_ids"])
            ids[i, :k] = torch.tensor(r["input_ids"])
            labels[i, :k] = torch.tensor(r["labels"])
            attn[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": attn}


# ----------------------------------------------------------------------------- model

def check_geodesic_scalars(model, model_path: str) -> None:
    """Compare the GeodesicNorm scalars with the file.

    transformers re-initialises any key it recorded as a shape mismatch, and
    the training env learned the hard way that such memory can look plausible while
    holding garbage. This is a few hundred scalars, so it is free to check.
    """
    from safetensors import safe_open

    truth = {}
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(shard, "pt") as f:
            for k in f.keys():
                if ".geodesic_" in k and k.rsplit(".", 1)[-1] in ("scale", "bias"):
                    truth[k] = f.get_tensor(k).float().reshape(-1)[0].item()
    params = dict(model.named_parameters())
    bad = [k for k, v in truth.items()
           if k not in params or abs(params[k].float().reshape(-1)[0].item() - v) > max(1e-2, abs(v) * 0.01)]
    if bad:
        raise RuntimeError(f"{len(bad)}/{len(truth)} geodesic scalars do not match {model_path}, e.g. {bad[:3]}")
    if is_main():
        logger.info("load check: %d geodesic scalars match the checkpoint", len(truth))


def resolve_targets(model, leaf_names: list[str], exclude: list[str]) -> list[str]:
    import torch.nn as nn

    leaves = set(leaf_names)
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in leaves
             and not any(x in n for x in exclude)]
    if not names:
        raise SystemExit(f"no nn.Linear module matches lora.target_leaf_names={leaf_names}")
    if is_main():
        counts: dict[str, int] = {}
        for n in names:
            leaf = n.rsplit(".", 1)[-1]
            counts[leaf] = counts.get(leaf, 0) + 1
        missing = sorted(leaves - set(counts))
        logger.info("LoRA targets: %s", ", ".join(f"{k}x{v}" for k, v in sorted(counts.items())))
        if missing:
            logger.warning("LoRA target names that match nothing: %s", missing)
    return names


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO if is_main() else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(cfg["seed"])
    out = Path(cfg["output_dir"])
    if is_main():
        out.mkdir(parents=True, exist_ok=True)
        (out / "sft_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        logger.info("config:\n%s", yaml.safe_dump(cfg, sort_keys=False))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train = load_rows(cfg["train_files"])
    if cfg["max_train_samples"]:
        train = train.shuffle(seed=cfg["seed"]).select(range(min(cfg["max_train_samples"], len(train))))
    train = tokenize_dataset(train, tokenizer, cfg, "train")
    evals = None
    if cfg["eval_files"]:
        evals = load_rows(cfg["eval_files"])
        if cfg["max_eval_samples"]:
            evals = evals.select(range(min(cfg["max_eval_samples"], len(evals))))
        evals = tokenize_dataset(evals, tokenizer, cfg, "eval")

    # The model code picks its attention backend at import and prints it:
    # look for "Using attention implementation: fa3" in the log.
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"], trust_remote_code=True, dtype=torch.bfloat16)
    check_geodesic_scalars(model, cfg["model_path"])
    model.config.use_cache = False

    lc = cfg["lora"]
    peft_cfg = LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], use_rslora=lc["use_rslora"],
        target_modules=resolve_targets(model, lc["target_leaf_names"], lc["exclude"]),
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    if cfg["training"].get("gradient_checkpointing"):
        model.enable_input_require_grads()
    if is_main():
        model.print_trainable_parameters()

    targs = dict(cfg["training"])
    if evals is None:
        targs.pop("eval_strategy", None)
        targs.pop("eval_steps", None)
    args = TrainingArguments(output_dir=str(out / "checkpoints"), remove_unused_columns=False,
                             seed=cfg["seed"], **targs)

    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=evals,
                      data_collator=Collator(tokenizer.pad_token_id))
    last = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if last and is_main():
        logger.info("resuming from %s", last)
    trainer.train(resume_from_checkpoint=last)

    adapter_dir = out / "adapter"
    trainer.save_model(str(adapter_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(adapter_dir))
        if cfg["merge_when_done"]:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from merge_lora import merge

            merge(Path(cfg["model_path"]), adapter_dir, out / "merged")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
