"""Build a small in-domain (central banking / finance) en<->fr SFT set from the OPUS ECB corpus.

ECB v1 (https://opus.nlpl.eu/ECB-v1.php): 196k sentence pairs from the European Central Bank's
website and publications (monetary policy, legal acts, banking supervision, payments, statistics),
professionally translated. Source: ECB, compiled by Alberto Simoes; reproduction permitted with
the source acknowledged. Cite J. Tiedemann, 2012, Parallel Data, Tools and Interfaces in OPUS, LREC.

The raw corpus is Moses-tokenized and part of it is damaged by PDF extraction (lost accents,
stray diacritics, broken hyphens) or is boilerplate (navigation paths, file sizes, tables), so:
  * detokenize both sides (sacremoses), rejoin "ECB / 2001/9" -> "ECB/2001/9";
  * keep full sentences only: 60-400 chars, fr/en length ratio 0.8-1.6, starts with a capital,
    ends with . ; or :, mostly letters, the same numbers on both sides;
  * drop lines with URLs, HTML entities, PDF damage (no accent at all in the French, U+00B4,
    Ø for é, soft hyphens, "celleci"-style lost hyphens, "--" for dashes), and duplicate
    English sources;
  * drop misaligned pairs: LaBSE (setu4993/LaBSE) cosine similarity between the two sides
    below MIN_SIM (0.80); this needs torch + transformers, and runs on a GPU if there is one;
  * hold out whole 500-line blocks for eval (neighbouring lines come from the same page), so no
    document is split between train and eval. Eval pairs whose English or French shares an
    8-word run with the selected train pairs are dropped too: ECB reuses formulas ("This Decision
    shall enter into force on...") that would otherwise reward memorising the train set.

Writes `messages` rows for sft/sft_lora.py, same prompt as the tutorial with domain "Finance",
alternating en->fr / fr->en, chat_template_kwargs {"reasoning_effort": "disabled"}:
  ecb_enfr_train.jsonl   N_TRAIN pairs (default 4000)
  ecb_enfr_eval.jsonl    N_EVAL pairs (default 400)

    pip install sacremoses   # plus torch and transformers for the LaBSE filter
    python make_ecb_enfr.py [OUT_DIR] [N_TRAIN] [N_EVAL]
"""
import io
import json
import random
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from sacremoses import MosesDetokenizer

URL = "https://object.pouta.csc.fi/OPUS-ECB/v1/moses/en-fr.txt.zip"
SYSTEM_PROMPT = (
    "You are a professional translation assistant. Your task is to translate text carefully, pay attention to the "
    "following:\n1. Preserve the casing of words.\n2. Maintain the inline tags as-is in the translated text.\n"
    "3. Follow the terminology mappings provided by the user.\n4. The domain of the text.\n"
    "Return only the translation, without any additional explanation."
)
DIRS = [("en", "fr", "English", "French"), ("fr", "en", "French", "English")]
DOMAIN = "Finance"
BLOCK, EVAL_BLOCK_SHARE, SEED = 500, 0.15, 0
LABSE, MIN_SIM = "setu4993/LaBSE", 0.80

ACCENT = re.compile(r"[éèêàçùâîôûëïÉÈÀÇ]")
BAD = re.compile(r"[´^�ŐØ\u00ad]|&[a-z]+;|https?:|www\.|@|\.pdf|\bpdf \d+ kB|\bkB\b|Navigation Path|--|\( ")
LOST_HYPHEN = re.compile(r"\b(?:celle|celui|ceux|celles)ci\b|\bci(?:dessus|dessous|après|joint)\b|c'estàdire|peutêtre")
NUM = re.compile(r"\d+")
SLASH = re.compile(r"\b([A-Z]{2,}) / (\d)")


def numbers(s: str) -> list[str]:
    return sorted(NUM.findall(re.sub(r"(?<=\d)[ ,.  ](?=\d)", "", s)))


def clean(en: str, fr: str) -> bool:
    if not (60 <= len(en) <= 400 and 60 <= len(fr) <= 500 and 0.8 <= len(fr) / len(en) <= 1.6):
        return False
    if BAD.search(en) or BAD.search(fr) or LOST_HYPHEN.search(fr) or not ACCENT.search(fr):
        return False
    if not (en[0].isupper() and fr[0].isupper() and en[-1] in ".;:" and fr[-1] in ".;:"):
        return False
    if sum(c.isalpha() for c in fr) < 0.75 * len(fr):  # tables, lists, references
        return False
    return numbers(en) == numbers(fr)


def labse_sim(en: list[str], fr: list[str], batch: int = 256) -> list[float]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(LABSE)
    model = AutoModel.from_pretrained(LABSE).to(dev).eval()

    @torch.no_grad()
    def emb(texts):
        x = tok(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
        return torch.nn.functional.normalize(model(**x).pooler_output, dim=-1)

    sims = []
    for i in range(0, len(en), batch):
        sims += (emb(en[i:i + batch]) * emb(fr[i:i + batch])).sum(-1).tolist()
    return sims


def ngrams(s: str, n: int = 8) -> set[str]:
    w = re.findall(r"\w+", s.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def row(en: str, fr: str, i: int) -> dict:
    text = {"en": en, "fr": fr}
    s, t, name_s, name_t = DIRS[i % 2]
    request = {"source_language": name_s, "target_language": name_t, "source_text": text[s],
               "terminology_mapping": [], "domain": DOMAIN}
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=2)},
                         {"role": "assistant", "content": text[t]}],
            "chat_template_kwargs": {"reasoning_effort": "disabled"}}


def main(out: Path, n_train: int, n_eval: int) -> None:
    print(f"downloading {URL}")
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
    en_lines = z.read("ECB.en-fr.en").decode().splitlines()
    fr_lines = z.read("ECB.en-fr.fr").decode().splitlines()
    detok = {"en": MosesDetokenizer("en"), "fr": MosesDetokenizer("fr")}

    rng = random.Random(SEED)
    n_blocks = (len(en_lines) + BLOCK - 1) // BLOCK
    eval_blocks = set(rng.sample(range(n_blocks), int(n_blocks * EVAL_BLOCK_SHARE)))
    pairs = {"train": [], "eval": []}
    seen = {"train": set(), "eval": set()}
    for i, (en, fr) in enumerate(zip(en_lines, fr_lines)):
        en = SLASH.sub(r"\1/\2", detok["en"].detokenize(en.split()))
        fr = SLASH.sub(r"\1/\2", detok["fr"].detokenize(fr.split()))
        if not clean(en, fr):
            continue
        split = "eval" if i // BLOCK in eval_blocks else "train"
        key = re.sub(r"\W+", "", en.lower())
        if key in seen[split]:
            continue
        seen[split].add(key)
        pairs[split].append((key, en, fr))
    pairs["eval"] = [p for p in pairs["eval"] if p[0] not in seen["train"]]
    print(f"{len(en_lines)} pairs -> clean, deduplicated: {len(pairs['train'])} train, {len(pairs['eval'])} eval")
    for split in pairs:
        sims = labse_sim([p[1] for p in pairs[split]], [p[2] for p in pairs[split]])
        pairs[split] = [p for p, s in zip(pairs[split], sims) if s >= MIN_SIM]
    print(f"aligned (LaBSE >= {MIN_SIM}): {len(pairs['train'])} train, {len(pairs['eval'])} eval")

    for split in pairs:
        rng.shuffle(pairs[split])
    pairs["train"] = pairs["train"][:n_train]
    train_grams = set().union(*(ngrams(en) | ngrams(fr) for _, en, fr in pairs["train"]))
    pairs["eval"] = [p for p in pairs["eval"] if not (ngrams(p[1]) | ngrams(p[2])) & train_grams]
    print(f"eval with no 8-word overlap with the {len(pairs['train'])} train pairs: {len(pairs['eval'])}")

    out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", n_train), ("eval", n_eval)):
        path = out / f"ecb_enfr_{split}.jsonl"
        with open(path, "w") as f:
            for i, (_, en, fr) in enumerate(pairs[split][:n]):
                f.write(json.dumps(row(en, fr, i), ensure_ascii=False) + "\n")
        print(f"{path}: {min(n, len(pairs[split]))} rows")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "."),
         int(sys.argv[2]) if len(sys.argv) > 2 else 4000,
         int(sys.argv[3]) if len(sys.argv) > 3 else 400)
