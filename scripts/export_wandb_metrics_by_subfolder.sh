#!/usr/bin/env bash
set -euo pipefail

# Export one CSV set per direct subfolder under a local W&B log root.
#
# Usage:
#   bash scripts/export_wandb_metrics_by_subfolder.sh \
#     --log-root log_wandb/pi0fast_libero \
#     --save-root log_trans_csv_by_subfolder \
#     --benchmark pi0fast_libero_v4 \
#     --meta v2

LOG_ROOT="log_wandb/pi0fast_libero"
SAVE_ROOT="log_trans_csv_by_subfolder"
BENCHMARK="pi0fast_libero_v4"
META="v2"

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
    --benchmark)
      BENCHMARK="$2"
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

exported=0
for subdir in "$LOG_ROOT"/*; do
  [[ -d "$subdir" ]] || continue

  if ! find "$subdir" -maxdepth 1 -type d -name "run-*" | grep -q .; then
    echo "[skip] $(basename "$subdir"): no run-* folders"
    continue
  fi

  name="$(basename "$subdir")"
  out_dir="$SAVE_ROOT/$name"
  mkdir -p "$out_dir"

  echo "[export] $name"
  PYTHONPATH=. python scripts/get_wandb_metrics.py \
    --meta "$META" \
    --benchmark "$BENCHMARK" \
    --log_root "$subdir" \
    --save_root "$out_dir"

  exported=1
done

if [[ "$exported" -eq 0 ]]; then
  echo "No valid subfolders with run-* found under: $LOG_ROOT" >&2
  exit 1
fi

echo "Done. CSVs saved under: $SAVE_ROOT"

