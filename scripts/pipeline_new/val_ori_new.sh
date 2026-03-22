#!/bin/bash

set -euo pipefail

# Summarize validation-set ori metrics and alpha Pareto fronts from eval/ori_logs.json.
#
# Usage:
#   scripts/pipeline_new/val_ori_new.sh
#   scripts/pipeline_new/val_ori_new.sh log_ckpt_new
#   scripts/pipeline_new/val_ori_new.sh log_ckpt_new /tmp/val_ori_new

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 [logs_dir] [save_dir]" >&2
    exit 1
fi

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

if [ "$#" -ge 1 ]; then
    logs_dir="$(realpath "$1")"
else
    logs_dir="$(realpath "$(resolve_default_logs_dir)")"
fi

if [ ! -d "${logs_dir}" ]; then
    echo "Directory not found: ${logs_dir}" >&2
    exit 1
fi

if [ "$#" -ge 2 ]; then
    save_dir="$(realpath "$2")"
else
    save_dir="${logs_dir}/pipeline_val_ori_new"
fi

cmd=(
    python -m failure_prob.pipeline.val_ori_new
    --logs-dir "${logs_dir}"
    --save-dir "${save_dir}"
)

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

env PYTHONPATH=. "${cmd[@]}"
