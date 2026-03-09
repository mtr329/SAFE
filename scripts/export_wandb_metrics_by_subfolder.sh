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
#   log_wandb/<dataset>/<model>/run-*
# Example:
#   log_wandb/pi0fast_libero/trans_best_0/run-...
#   log_wandb/openvla/ref/run-...
#   log_wandb/pi0_libero/trans0/run-...

LOG_ROOT="log_wandb"
SAVE_ROOT="log_trans_csv_by_subfolder"
META="v2"
DATASET_FILTER=""
MODEL_FILTER=""

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

# Map log_wandb dataset folder -> benchmark key in scripts/get_wandb_metrics.py
benchmark_from_dataset() {
  case "$1" in
    pi0fast_libero) echo "pi0fast_libero_v4" ;;
    pi0_libero) echo "pi0diff_libero_v1" ;;
    openvla) echo "openvla_libero_v2" ;;
    opi0_simpler) echo "opi0_simpler_v1" ;;
    *)
      return 1
      ;;
  esac
}

exported=0
failed=0
failed_items=()
for dataset_dir in "$LOG_ROOT"/*; do
  [[ -d "$dataset_dir" ]] || continue
  dataset_name="$(basename "$dataset_dir")"

  if [[ -n "$DATASET_FILTER" && "$dataset_name" != "$DATASET_FILTER" ]]; then
    continue
  fi

  if ! benchmark="$(benchmark_from_dataset "$dataset_name")"; then
    echo "[skip] $dataset_name: no benchmark mapping"
    continue
  fi

  for model_dir in "$dataset_dir"/*; do
    [[ -d "$model_dir" ]] || continue
    model_name="$(basename "$model_dir")"

    if [[ -n "$MODEL_FILTER" && "$model_name" != "$MODEL_FILTER" ]]; then
      continue
    fi

    if ! find "$model_dir" -maxdepth 1 -type d -name "run-*" | grep -q .; then
      echo "[skip] ${dataset_name}/${model_name}: no run-* folders"
      continue
    fi

    out_dir="$SAVE_ROOT/$dataset_name/$model_name"
    mkdir -p "$out_dir"

    echo "[export] ${dataset_name}/${model_name} -> benchmark=${benchmark}"
    if ! PYTHONPATH=. python scripts/get_wandb_metrics.py \
      --meta "$META" \
      --benchmark "$benchmark" \
      --log_root "$model_dir" \
      --save_root "$out_dir"; then
      echo "[warn] export failed for ${dataset_name}/${model_name}, continue..."
      failed=$((failed + 1))
      failed_items+=("${dataset_name}/${model_name}")
      continue
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
