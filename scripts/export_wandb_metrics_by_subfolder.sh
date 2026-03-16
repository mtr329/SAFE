#!/usr/bin/env bash
set -euo pipefail

# Export CSVs by dataset/model subfolders under a local W&B log root.
#
# Usage:
#   bash scripts/export_wandb_metrics_by_subfolder.sh \
#     --log-root log_wandb \
#     --save-root log_trans_csv_by_subfolder \
#     --meta v2
#
# Expected log structure:
#   log_wandb/<dataset>/<subfolder>/wandb/run-*
# Example:
#   log_wandb/pizero_fast/indep/wandb/run-...
#   log_wandb/pizero_fast/embed_cosine/wandb/run-...
#   log_wandb/openvla/lstm/wandb/run-...

LOG_ROOT="log_wandb"
SAVE_ROOT="log_trans_csv_by_subfolder"
META="v2"
DATASET_FILTER=""
MODEL_FILTER=""
METRIC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --save-root)
      SAVE_ROOT="$2"
      shift 2
      ;;
    --dataset)
      DATASET_FILTER="$2"
      shift 2
      ;;
    --model)
      MODEL_FILTER="$2"
      shift 2
      ;;
    --meta)
      META="$2"
      shift 2
      ;;
    --metric)
      METRIC="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$LOG_ROOT" ]]; then
  echo "log root not found: $LOG_ROOT" >&2
  exit 1
fi

mkdir -p "$SAVE_ROOT"

exported=0
failed=0
failed_items=()
for dataset_dir in "$LOG_ROOT"/*; do
  [[ -d "$dataset_dir" ]] || continue
  dataset_name="$(basename "$dataset_dir")"

  if [[ -n "$DATASET_FILTER" && "$dataset_name" != "$DATASET_FILTER" ]]; then
    continue
  fi

  for model_dir in "$dataset_dir"/*; do
    [[ -d "$model_dir" ]] || continue
    model_name="$(basename "$model_dir")"

    if [[ -n "$MODEL_FILTER" && "$model_name" != "$MODEL_FILTER" ]]; then
      continue
    fi

    if ! find "$model_dir" -type d -name "run-*" | grep -q .; then
      echo "[skip] ${dataset_name}/${model_name}: no run-* folders under subdirectory"
      continue
    fi

    out_dir="$SAVE_ROOT/$dataset_name/$model_name"
    mkdir -p "$out_dir"

    echo "[export] ${dataset_name}/${model_name} -> benchmark=auto"
    cmd=(python scripts/get_wandb_metrics.py
      --meta "$META"
      --benchmark auto
      --log_root "$model_dir"
      --save_root "$out_dir")
    if [[ -n "$METRIC" ]]; then
      cmd+=(--metric "$METRIC")
    fi

    if ! env WANDB_MODE=offline PYTHONPATH=. "${cmd[@]}"; then
      echo "[warn] export failed for ${dataset_name}/${model_name}, continue..."
      failed=$((failed + 1))
      failed_items+=("${dataset_name}/${model_name}")
      continue
    fi

    if [[ -n "$METRIC" ]]; then
      metric_printed=0
      for benchmark_csv in "$out_dir"/*.csv; do
        [[ -f "$benchmark_csv" ]] || continue
        benchmark_base="$(basename "$benchmark_csv")"
        if [[ "$benchmark_base" == *-* ]]; then
          continue
        fi
        echo "[metric] ${dataset_name}/${model_name} metric=${METRIC} csv=${benchmark_base}"
        python -c "import pandas as pd; p='$benchmark_csv'; df=pd.read_csv(p); cols=['method','model.name','val_seen']; keep=[c for c in cols if c in df.columns]; print(df[keep].to_string(index=False))"
        metric_printed=1
      done
      if [[ "$metric_printed" -eq 0 ]]; then
        echo "[metric] ${dataset_name}/${model_name}: benchmark csv not found for metric print"
      fi
    fi

    exported=1
  done
done

if [[ "$exported" -eq 0 ]]; then
  echo "No valid dataset/model subfolders with run-* found under: $LOG_ROOT" >&2
  exit 1
fi

echo "Done. CSVs saved under: $SAVE_ROOT"
if [[ "$failed" -gt 0 ]]; then
  echo "[warn] ${failed} exports failed:"
  for item in "${failed_items[@]}"; do
    echo "  - $item"
  done
fi
