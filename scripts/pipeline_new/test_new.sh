#!/bin/bash

set -euo pipefail

# Run test-stage comparison using the selection exported by val_new.py.
#
# Usage:
#   scripts/pipeline_new/test_new.sh <selection_path>
#   scripts/pipeline_new/test_new.sh log_ckpt <selection_path>
#   scripts/pipeline_new/test_new.sh log_ckpt <selection_path> /tmp/test_new

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 [logs_dir] <selection_path> [save_dir]" >&2
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
    logs_dir="$(realpath "$(resolve_default_logs_dir)")"
    selection_path="$(realpath "$1")"
    save_dir="${logs_dir}/pipeline_test_new"
elif [ "$#" -eq 2 ]; then
    logs_dir="$(realpath "$1")"
    selection_path="$(realpath "$2")"
    save_dir="${logs_dir}/pipeline_test_new"
else
    logs_dir="$(realpath "$1")"
    selection_path="$(realpath "$2")"
    save_dir="$(realpath "$3")"
fi

if [ ! -d "${logs_dir}" ]; then
    echo "Directory not found: ${logs_dir}" >&2
    exit 1
fi

if [ ! -f "${selection_path}" ]; then
    echo "Selection file not found: ${selection_path}" >&2
    exit 1
fi

cmd=(
    python -m failure_prob.pipeline.test_new
    --logs-dir "${logs_dir}"
    --selection-path "${selection_path}"
    --save-dir "${save_dir}"
)

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

env PYTHONPATH=. "${cmd[@]}"
