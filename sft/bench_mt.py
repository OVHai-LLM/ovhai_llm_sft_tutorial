"""Small en<->fr translation benchmark: BLEU and chrF++ per direction, greedy decoding.

    python bench_mt.py --model-dir CKPT --label base --out bench_base.json
    python bench_mt.py --compare bench_base.json bench_finetuned.json

Prompt and decoding match the Colab tutorial (colab/): the same system prompt and JSON
request, reasoning_effort="disabled", greedy, one sentence at a time (no padding, so nothing
can leak into the Mamba state), answer taken from the final channel.

The eval file is jsonl in either shape:
  * {"source_language", "target_language", "source", "reference", "domain"?}
  * {"messages": [system, user (the JSON request), assistant (the reference)]}
      (open_data/ecb_enfr_eval.jsonl, the default in the sft image; opus100_enfr_eval.jsonl)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a professional translation assistant. Your task is to translate text carefully, pay attention to the "
    "following:\n1. Preserve the casing of words.\n2. Maintain the inline tags as-is in the translated text.\n"
    "3. Follow the terminology mappings provided by the user.\n4. The domain of the text.\n"
    "Return only the translation, without any additional explanation."
)
PAIRS = [("English", "French"), ("French", "English")]


def load_eval(path: str, max_n: int | None) -> list[dict]:
    rows = []
    for line in open(path):
        r = json.loads(line)
        if "messages" in r:
            req = json.loads(next(m["content"] for m in r["messages"] if m["role"] == "user"))
            ref = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
            r = {"source_language": req["source_language"], "target_language": req["target_language"],
                 "source": req["source_text"], "reference": ref, "domain": req.get("domain", "General")}
        rows.append(r)
    if max_n:  # keep both directions balanced
        per = max_n // len(PAIRS)
        rows = [r for src, _ in PAIRS for r in [x for x in rows if x["source_language"] == src][:per]]
    return rows


def build_messages(text: str, src: str, tgt: str, domain: str = "General") -> list[dict]:
    request = {"source_language": src, "target_language": tgt, "source_text": text,
               "terminology_mapping": [], "domain": domain}
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False, indent=2)}]


def run(args) -> None:
    import sacrebleu
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_eval(args.eval_file, args.max_n)
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()

    t0 = time.time()
    hyps = []
    with torch.no_grad():
        for r in tqdm(rows, desc=f"bench {args.label}"):
            inputs = tok.apply_chat_template(
                build_messages(r["source"], r["source_language"], r["target_language"], r.get("domain", "General")),
                add_generation_prompt=True, reasoning_effort="disabled",
                return_tensors="pt", return_dict=True).to("cuda")
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
            answer = tok.decode(out[0, inputs["input_ids"].shape[1]:])
            hyps.append(answer.split("<|content|>")[-1].split("<|channel_end|>")[0].strip())

    scores = {}
    for src, tgt in PAIRS:
        idx = [i for i, r in enumerate(rows) if r["source_language"] == src]
        if not idx:
            continue
        h, refs = [hyps[i] for i in idx], [[rows[i]["reference"] for i in idx]]
        scores[f"{src[:2].lower()}->{tgt[:2].lower()}"] = {
            "bleu": round(sacrebleu.corpus_bleu(h, refs).score, 2),
            "chrf++": round(sacrebleu.corpus_chrf(h, refs, word_order=2).score, 2),
            "n": len(idx)}
    result = {"label": args.label, "model_dir": str(args.model_dir), "eval_file": args.eval_file,
              "n": len(rows), "seconds": round(time.time() - t0), "scores": scores,
              "samples": [{**r, "hypothesis": h} for r, h in zip(rows, hyps)]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for pair, s in scores.items():
        print(f"{args.label} {pair}: BLEU {s['bleu']:.1f} | chrF++ {s['chrf++']:.1f} (n={s['n']})")
    print(f"wrote {args.out}")


def compare(paths: list[str]) -> None:
    res = [json.loads(Path(p).read_text()) for p in paths]
    pairs = list(res[0]["scores"])
    head = f"{'':<8}" + "".join(f"{r['label'] + ' BLEU':>18}{r['label'] + ' chrF++':>20}" for r in res)
    print(f"benchmark: {res[0]['eval_file']} ({res[0]['n']} sentences)\n{head}")
    for pair in pairs:
        print(f"{pair:<8}" + "".join(f"{r['scores'][pair]['bleu']:>18.1f}{r['scores'][pair]['chrf++']:>20.1f}"
                                     for r in res))
    a, b = res[0]["samples"], res[-1]["samples"]
    for i in [0, len(a) - 1]:
        print(f"\n{a[i]['source_language']} -> {a[i]['target_language']}\n   {a[i]['source']}\n"
              f"   reference:           {a[i]['reference']}\n"
              f"   {res[0]['label']:<20} {a[i]['hypothesis']}\n   {res[-1]['label']:<20} {b[i]['hypothesis']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir")
    p.add_argument("--label", default="model")
    p.add_argument("--eval-file", default=os.environ.get("BENCH_FILE", "/opt/mt/data/open/ecb_enfr_eval.jsonl"))
    p.add_argument("--max-n", type=int, default=None, help="sentences to use, split evenly between directions")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--out")
    p.add_argument("--compare", nargs="+", metavar="JSON", help="print a table from bench JSON files")
    a = p.parse_args()
    if a.compare:
        compare(a.compare)
    elif a.model_dir and a.out:
        run(a)
    else:
        sys.exit("need --model-dir and --out, or --compare")
