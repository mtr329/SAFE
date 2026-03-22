#!/bin/bash

set -euo pipefail

# Run validation selection first, then evaluate the selected configs on test logs.
#
# Usage:
#   scripts/pipeline_new/run_all.sh
#   scripts/pipeline_new/run_all.sh log_ckpt
#   scripts/pipeline_new/run_all.sh log_ckpt /tmp/pipeline_new_summary

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
    save_root="${logs_dir}/pipeline_new_summary"
fi

val_save_dir="${save_root}/val"
test_save_dir="${save_root}/test"
selection_path="${val_save_dir}/val_selection.json"

cmd_val=(
    bash scripts/pipeline_new/val_new.sh
    "${logs_dir}"
    "${val_save_dir}"
)

printf 'Running:'
printf ' %q' "${cmd_val[@]}"
printf '\n'
env PYTHONPATH=. "${cmd_val[@]}"

cmd_test=(
    bash scripts/pipeline_new/test_new.sh
    "${logs_dir}"
    "${selection_path}"
    "${test_save_dir}"
)

printf 'Running:'
printf ' %q' "${cmd_test[@]}"
printf '\n'
env PYTHONPATH=. "${cmd_test[@]}"

echo "Validation summary: ${val_save_dir}"
echo "Test summary: ${test_save_dir}"
