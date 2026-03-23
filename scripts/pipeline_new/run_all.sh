#!/bin/bash

set -euo pipefail

# Run validation first, then select checkpoints from val outputs and run test eval.
#
# Usage:
#   scripts/pipeline_new/run_all.sh
#   scripts/pipeline_new/run_all.sh log_ckpt_new/pizero_fast
#   scripts/pipeline_new/run_all.sh --gpu 0 log_ckpt_new/pizero_fast /tmp/pipeline_new_summary

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
save_root=""
extra_test_args=()

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
        --force-eval)
            extra_test_args+=("$1")
            shift
            ;;
        --min-curve-fraction|--grid-step|--penalty-det-time)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for $1" >&2
                exit 1
            fi
            extra_test_args+=("$1" "$2")
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
            elif [ -z "${save_root}" ]; then
                save_root="$1"
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

if [ -n "${save_root}" ]; then
    save_root="$(realpath "${save_root}")"
else
    save_root="${logs_dir}/pipeline_new_summary"
fi

val_save_dir="${save_root}/val"
test_save_dir="${save_root}/test"

cmd_val=(
    bash scripts/pipeline_new/val_new.sh
)
if [ -n "${gpu_id}" ]; then
    cmd_val+=(--gpu "${gpu_id}")
fi
cmd_val+=("${logs_dir}" "${val_save_dir}")

printf 'Running:'
printf ' %q' "${cmd_val[@]}"
printf '\n'
env PYTHONPATH=. "${cmd_val[@]}"

cmd_test=(
    bash scripts/pipeline_new/test_new.sh
)
if [ -n "${gpu_id}" ]; then
    cmd_test+=(--gpu "${gpu_id}")
fi
cmd_test+=("${logs_dir}" "${test_save_dir}" "${extra_test_args[@]}")

printf 'Running:'
printf ' %q' "${cmd_test[@]}"
printf '\n'
env PYTHONPATH=. "${cmd_test[@]}"

echo "Validation summary: ${val_save_dir}"
echo "Test summary: ${test_save_dir}"
