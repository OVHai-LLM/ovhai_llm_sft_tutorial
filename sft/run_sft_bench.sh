#!/usr/bin/env bash
# Benchmark the base checkpoint, LoRA-SFT it, benchmark the merged result, print both.
# In the showcase image (ovhaillm-mt-tutorial): `run_sft_bench.sh`, or ovh/launch_sft_baked.sh.
#
#   run_sft_bench.sh [key=value overrides for sft_lora.py ...]
#   BENCH_N=50 run_sft_bench.sh training.max_steps=20          # quick check
#
# Writes to $OUT (default /workspace/out):
#   bench_base.json, adapter/, merged/, checkpoints/, sft_config.yaml, bench_finetuned.json, bench.txt
# A rerun skips the base benchmark if bench_base.json is already there, and training resumes
# from the newest checkpoint.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG=${CONFIG:-${SFT_CONFIG:-configs/mt_lora_ecb.yaml}}
OUT=${OUT:-/workspace/out}
BASE=${BASE:-/opt/ckpt/ovhaillm-rl}     # must match model_path in $CONFIG
BENCH_N=${BENCH_N:-}                         # empty = the whole eval file
BENCH_ARGS=(${BENCH_N:+--max-n "$BENCH_N"})
mkdir -p "$OUT"

if [ ! -s "$OUT/bench_base.json" ]; then
    python bench_mt.py --model-dir "$BASE" --label base --out "$OUT/bench_base.json" "${BENCH_ARGS[@]}"
fi
./train.sh "$CONFIG" model_path="$BASE" output_dir="$OUT" merge_when_done=true "$@"
python bench_mt.py --model-dir "$OUT/merged" --label finetuned --out "$OUT/bench_finetuned.json" "${BENCH_ARGS[@]}"
python bench_mt.py --compare "$OUT/bench_base.json" "$OUT/bench_finetuned.json" | tee "$OUT/bench.txt"
