#!/bin/bash

set -euo pipefail

# Evaluate checkpoints from training log directories.
#
# Usage:
#   scripts/eval/eval.sh
#   scripts/eval/eval.sh logs/pizero_fast-default-lstm-lstm/20260315/160125
#   scripts/eval/eval.sh logs/pizero_fast-default-lstm-lstm
#
# Default behavior:
#   Traverse logs/<method>/<date>/<run> and evaluate each run directory.
#
# Optional argument:
#   Restrict evaluation to the specified directory. If the directory itself
#   contains config.yaml, only that run is evaluated. Otherwise, all descendant
#   directories containing config.yaml are evaluated.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

run_eval() {
    local log_dir="$1"
    local config_path="${log_dir}/config.yaml"

    if [ ! -d "${log_dir}" ]; then
        echo "Skipping missing directory: ${log_dir}" >&2
        return
    fi

    if [ ! -f "${config_path}" ]; then
        echo "Skipping directory without config.yaml: ${log_dir}" >&2
        return
    fi

    local cmd=(
        python -m failure_prob.eval
        --config-path "${log_dir}"
        --config-name config
        "train.eval_ckpt_path=${log_dir}"
        "dataset.data_path_prefix="
    )

    printf 'Running:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    "${cmd[@]}"
}

collect_target_dirs() {
    local target_root="$1"

    if [ ! -d "${target_root}" ]; then
        echo "Directory not found: ${target_root}" >&2
        exit 1
    fi

    if [ -f "${target_root}/config.yaml" ]; then
        printf '%s\n' "${target_root}"
        return
    fi

    find "${target_root}" -type f -name config.yaml -printf '%h\n' | sort -u
}

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [log_dir]" >&2
    exit 1
fi

if [ "$#" -eq 1 ]; then
    target_root="$(realpath "$1")"
else
    target_root="$(realpath logs)"
fi

mapfile -t target_dirs < <(collect_target_dirs "${target_root}")

if [ "${#target_dirs[@]}" -eq 0 ]; then
    echo "No evaluation directories found under: ${target_root}" >&2
    exit 1
fi

for log_dir in "${target_dirs[@]}"; do
    run_eval "${log_dir}"
done
