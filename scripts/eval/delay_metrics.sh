#!/bin/bash

set -euo pipefail

# Summarize delay metrics from evaluation logs.
#
# Usage:
#   scripts/eval/delay_metrics.sh
#   DELAY_METRICS_LOGS_DIR=log_ckpt scripts/eval/delay_metrics.sh
#   scripts/eval/delay_metrics.sh log_ckpt/pizero_fast-default-lstm-lstm
#   scripts/eval/delay_metrics.sh log_ckpt/pizero_fast-default-lstm-lstm/20260315/160125

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [logs_dir]" >&2
    exit 1
fi

resolve_default_logs_dir() {
    if [ -n "${DELAY_METRICS_LOGS_DIR:-}" ]; then
        printf '%s\n' "${DELAY_METRICS_LOGS_DIR}"
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
    python -m failure_prob.mrefine.delay_metrics
    "${logs_dir}"
)

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

env PYTHONPATH=. "${cmd[@]}"
