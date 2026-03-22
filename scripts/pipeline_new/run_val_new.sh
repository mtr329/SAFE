#!/bin/bash

set -euo pipefail

# Run the validation pipeline for trained checkpoints:
# 1. batch eval to produce eval/*.json
# 2. summarize validation-set ori metrics by seed
# 3. summarize validation-set new/t@bal_acc metrics by seed
#
# Usage:
#   scripts/pipeline_new/run_val_new.sh
#   scripts/pipeline_new/run_val_new.sh log_ckpt_new
#   scripts/pipeline_new/run_val_new.sh log_ckpt_new /tmp/pipeline_val_eval_new

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 [logs_dir] [save_root]" >&2
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
    save_root="$(realpath "$2")"
else
    save_root="${logs_dir}/pipeline_val_eval_new"
fi

ori_save_dir="${save_root}/ori"
new_save_dir="${save_root}/new"

cmd_eval=(
    bash scripts/pipeline_new/eval_new.sh
    "${logs_dir}"
)
cmd_ori=(
    bash scripts/pipeline_new/val_ori_new.sh
    "${logs_dir}"
    "${ori_save_dir}"
)
cmd_new=(
    bash scripts/pipeline_new/val_new.sh
    "${logs_dir}"
    "${new_save_dir}"
)

printf 'Running:'
printf ' %q' "${cmd_eval[@]}"
printf '\n'
env PYTHONPATH=. "${cmd_eval[@]}"

printf 'Running:'
printf ' %q' "${cmd_ori[@]}"
printf '\n'
env PYTHONPATH=. "${cmd_ori[@]}"

printf 'Running:'
printf ' %q' "${cmd_new[@]}"
printf '\n'
env PYTHONPATH=. "${cmd_new[@]}"

echo "Ori validation summary: ${ori_save_dir}"
echo "New validation summary: ${new_save_dir}"
