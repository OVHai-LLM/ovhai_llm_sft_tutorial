"""Build a small open en<->fr SFT set from opus-100 (Helsinki-NLP/opus-100, en-fr).

Same filter, prompt and direction alternation as the Colab tutorial (colab/), written as
`messages` rows for sft/sft_lora.py:
  opus100_enfr_train.jsonl   2000 pairs from the train split (shuffle seed 0)
  opus100_enfr_eval.jsonl     200 pairs from the validation split
Half en->fr, half fr->en. Each row also carries chat_template_kwargs
{"reasoning_effort": "disabled"}, as in the tutorial.

    python make_opus100_enfr.py [OUT_DIR] [N_TRAIN] [N_EVAL]
"""
import json
import sys
from pathlib import Path

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a professional translation assistant. Your task is to translate text carefully, pay attention to the "
    "following:\n1. Preserve the casing of words.\n2. Maintain the inline tags as-is in the translated text.\n"
    "3. Follow the terminology mappings provided by the user.\n4. The domain of the text.\n"
    "Return only the translation, without any additional explanation."
)
DIRS = [("en", "fr", "English", "French"), ("fr", "en", "French", "English")]


def is_clean(r):
    en, fr = r["translation"]["en"], r["translation"]["fr"]
    return 20 <= len(en) <= 500 and 20 <= len(fr) <= 500 and 0.5 <= len(fr) / len(en) <= 2


def row(r, i):
    s, t, name_s, name_t = DIRS[i % 2]
    request = {"source_language": name_s, "target_language": name_t, "source_text": r["translation"][s],
               "terminology_mapping": [], "domain": "General"}
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=2)},
                         {"role": "assistant", "content": r["translation"][t]}],
            "chat_template_kwargs": {"reasoning_effort": "disabled"}}


def write(split, n, path):
    ds = load_dataset("Helsinki-NLP/opus-100", "en-fr", split=split).shuffle(seed=0)
    ds = ds.select(range(min(len(ds), 4 * n))).filter(is_clean).select(range(n))
    with open(path, "w") as f:
        for i, r in enumerate(ds):
            f.write(json.dumps(row(r, i), ensure_ascii=False) + "\n")
    print(f"{path}: {n} rows")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    n_eval = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    out.mkdir(parents=True, exist_ok=True)
    write("train", n_train, out / "opus100_enfr_train.jsonl")
    write("validation", n_eval, out / "opus100_enfr_eval.jsonl")
