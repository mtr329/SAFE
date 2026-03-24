#!/bin/bash

set -euo pipefail

# Run train/validation-only metric computation for trained checkpoints.
# This writes val/ori_logs.json and val/new_logs.json under each run directory,
# plus per-method summaries under <save_dir>/methods/.
#
# Usage:
#   scripts/pipeline_new/val_new.sh
#   scripts/pipeline_new/val_new.sh --logs-dir log_ckpt_new/pizero_fast
#   scripts/pipeline_new/val_new.sh --gpu 0 --logs-dir log_ckpt_new/pizero_fast --save-dir /tmp/val_new
#   scripts/pipeline_new/val_new.sh --logs-dir log_ckpt_new/pizero_fast --method trans

resolve_default_logs_dir() {
    if [ -n "${PIPELINE_NEW_LOGS_DIR:-}" ]; then
        printf '%s\n' "${PIPELINE_NEW_LOGS_DIR}"
        return
    fi

    for candidate in log_ckpt_new log_ckpt logs; do
        if [ -d "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    echo "Could not find a default logs directory. Tried: log_ckpt_new, log_ckpt, logs" >&2
    exit 1
}

gpu_id="${CUDA_VISIBLE_DEVICES:-}"
logs_dir=""
save_dir=""
method_name=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --gpu)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --gpu" >&2
                exit 1
            fi
            gpu_id="$2"
            shift 2
            ;;
        --logs-dir)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --logs-dir" >&2
                exit 1
            fi
            logs_dir="$2"
            shift 2
            ;;
        --save-dir)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --save-dir" >&2
                exit 1
            fi
            save_dir="$2"
            shift 2
            ;;
        --method)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --method" >&2
                exit 1
            fi
            method_name="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '1,12p' "$0"
            exit 0
            ;;
        *)
            if [ -z "${logs_dir}" ]; then
                logs_dir="$1"
                shift
            elif [ -z "${save_dir}" ]; then
                save_dir="$1"
                shift
            else
                echo "Unknown argument: $1" >&2
                exit 1
            fi
            ;;
    esac
done

if [ -n "${logs_dir}" ]; then
    logs_dir="$(realpath "${logs_dir}")"
else
    logs_dir="$(realpath "$(resolve_default_logs_dir)")"
fi

if [ ! -d "${logs_dir}" ]; then
    echo "Directory not found: ${logs_dir}" >&2
    exit 1
fi

if [ -n "${save_dir}" ]; then
    save_dir="$(realpath "${save_dir}")"
else
    save_dir="${logs_dir}/pipeline_val_new"
fi

cmd=(
    python -m failure_prob.pipeline.val_new
    --logs-dir "${logs_dir}"
    --save-dir "${save_dir}"
)

if [ -n "${method_name}" ]; then
    cmd+=(--method "${method_name}")
fi

printf 'Running:'
if [ -n "${gpu_id}" ]; then
    printf ' %q' "CUDA_VISIBLE_DEVICES=${gpu_id}"
fi
printf ' %q' "${cmd[@]}"
printf '\n'

if [ -n "${gpu_id}" ]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" env PYTHONPATH=. "${cmd[@]}"
else
    env PYTHONPATH=. "${cmd[@]}"
fi
