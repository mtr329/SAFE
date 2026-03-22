#!/bin/bash

set -euo pipefail

# Evaluate all trained checkpoints under the target logs directory.
#
# Usage:
#   scripts/pipeline_new/eval_new.sh
#   scripts/pipeline_new/eval_new.sh log_ckpt_new

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [logs_dir]" >&2
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

if [ "$#" -eq 1 ]; then
    logs_dir="$(realpath "$1")"
else
    logs_dir="$(realpath "$(resolve_default_logs_dir)")"
fi

if [ ! -d "${logs_dir}" ]; then
    echo "Directory not found: ${logs_dir}" >&2
    exit 1
fi

cmd=(
    bash scripts/eval/eval.sh
    "${logs_dir}"
)

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

env PYTHONPATH=. "${cmd[@]}"
