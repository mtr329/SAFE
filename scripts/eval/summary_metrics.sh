#!/bin/bash

set -euo pipefail

# Summarize ori, delay, new, and ref metrics from evaluation logs.
#
# Usage:
#   scripts/eval/summary_metrics.sh
#   SUMMARY_METRICS_LOGS_DIR=log_ckpt scripts/eval/summary_metrics.sh
#   scripts/eval/summary_metrics.sh log_ckpt/pizero_fast-default-lstm-lstm
#   scripts/eval/summary_metrics.sh log_ckpt/pizero_fast-default-lstm-lstm/20260315/160125
#   scripts/eval/summary_metrics.sh log_ckpt /tmp/my_summary

if [ "$#" -gt 2 ]; then
    echo "Usage: $0 [logs_dir] [save_dir]" >&2
    exit 1
fi

resolve_default_logs_dir() {
    if [ -n "${SUMMARY_METRICS_LOGS_DIR:-}" ]; then
        printf '%s\n' "${SUMMARY_METRICS_LOGS_DIR}"
        return
    fi

    for candidate in log_ckpt logs; do
        if [ -d "${candidate}" ]; then
            printf '%s\n' "${candidate}"
            return
        fi
    done

    echo "Could not find a default logs directory. Tried: log_ckpt, logs" >&2
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

save_dir=""
if [ "$#" -ge 2 ]; then
    save_dir="$(realpath "$2")"
fi

run_summary() {
    local module_name="$1"
    local cmd=(
        python -m "${module_name}"
        "${logs_dir}"
    )

    if [ -n "${save_dir}" ]; then
        cmd+=(
            --save-dir "${save_dir}"
        )
    fi

    printf 'Running:'
    printf ' %q' "${cmd[@]}"
    printf '\n'

    env PYTHONPATH=. "${cmd[@]}"
}

run_summary failure_prob.mrefine.ori_summary
run_summary failure_prob.mrefine.delay_summary
run_summary failure_prob.mrefine.new_summary
run_summary failure_prob.mrefine.ref_summary
