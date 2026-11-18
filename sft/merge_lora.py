#!/usr/bin/env python
"""Merge a LoRA adapter into an OvhaiLLM checkpoint, at the safetensors level.

    python sft/merge_lora.py BASE_DIR ADAPTER_DIR OUT_DIR

Why not peft's merge_and_unload + save_pretrained: the model is loaded in bf16,
which rounds the tensors the checkpoint stores in fp32 (mamba D/dt_bias, the
geodesic scalars, ...) and widens 0-dim scalars to shape (1,). Saving that model
would ship those changes. Here only the LoRA-targeted weights are rewritten,
W' = W + scale * B @ A computed in fp32 and cast back to W's dtype; every other
tensor, and every non-weight file (model code, tokenizer, chat templates), is
copied from the base unchanged.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

PREFIX = "base_model.model."


def load_deltas(adapter_dir: Path) -> tuple[dict, dict]:
    import torch
    from safetensors.torch import load_file

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    for unsupported in ("rank_pattern", "alpha_pattern"):
        if cfg.get(unsupported):
            raise NotImplementedError(f"{unsupported} is not supported by this merger")
    if cfg.get("use_dora"):
        raise NotImplementedError("DoRA is not supported by this merger")
    if cfg.get("modules_to_save"):
        raise NotImplementedError("modules_to_save is not supported by this merger")
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r

    weights = load_file(str(adapter_dir / "adapter_model.safetensors"))
    deltas = {}
    for key, a in weights.items():
        if ".lora_A." not in key:
            continue
        b = weights[key.replace(".lora_A.", ".lora_B.")]
        module = key.split(".lora_A.")[0].removeprefix(PREFIX)
        deltas[f"{module}.weight"] = (b.float() @ a.float()) * scale
    if not deltas:
        raise RuntimeError(f"no lora_A tensors in {adapter_dir}")
    return deltas, cfg


def merge(base_dir: Path, adapter_dir: Path, out_dir: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    deltas, cfg = load_deltas(adapter_dir)
    umask = os.umask(0)
    os.umask(umask)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = set(deltas)
    for path in sorted(base_dir.iterdir()):
        if path.is_dir():
            shutil.copytree(path, out_dir / path.name, dirs_exist_ok=True)
        elif path.suffix != ".safetensors":
            shutil.copy2(path, out_dir / path.name)
    for shard in sorted(base_dir.glob("*.safetensors")):
        tensors = {}
        with safe_open(str(shard), framework="pt") as f:
            meta = f.metadata() or {"format": "pt"}
            for key in f.keys():
                t = f.get_tensor(key)
                if key in deltas:
                    d = deltas[key]
                    if d.shape != t.shape:
                        raise RuntimeError(f"{key}: delta {tuple(d.shape)} vs weight {tuple(t.shape)}")
                    t = (t.float() + d).to(t.dtype)
                    pending.discard(key)
                tensors[key] = t
        save_file(tensors, str(out_dir / shard.name), metadata=meta)
        # save_file writes through a temp file and renames it, which leaves the shard 0600 --
        # unreadable to anyone but the training user. Give it the mode a plain open() would.
        os.chmod(out_dir / shard.name, 0o666 & ~umask)
    if pending:
        raise RuntimeError(f"{len(pending)} adapter targets not found in the base, e.g. {sorted(pending)[:3]}")
    (out_dir / "LORA_MERGE.txt").write_text(
        f"base:    {base_dir.resolve()}\nadapter: {adapter_dir.resolve()}\n"
        f"r={cfg['r']} alpha={cfg['lora_alpha']} rslora={cfg.get('use_rslora', False)} "
        f"targets={len(deltas)}\nmerged {datetime.now():%Y-%m-%d %H:%M:%S} by sft/merge_lora.py\n")
    print(f"merged {len(deltas)} LoRA weights into {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    merge(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
