#!/bin/bash

set -euo pipefail

ROOT="${1:-log_ckpt_new}"
APPLY=0

if [[ "${1:-}" == "--yes" ]]; then
    APPLY=1
    ROOT="${2:-log_ckpt_new}"
fi

if [[ ! -d "${ROOT}" ]]; then
    echo "Root directory not found: ${ROOT}" >&2
    exit 1
fi

mapfile -t TARGETS < <(
    find "${ROOT}" -type d \
        \( -regex ".*/seed[0-9]+/trans[^/]*" -o -regex ".*/pipeline_val_new/methods/trans[^/]*" \) \
        | sort
)

if [[ ${#TARGETS[@]} -eq 0 ]]; then
    echo "No matching directories found under ${ROOT}."
    exit 0
fi

echo "Matched directories:"
printf '  %s\n' "${TARGETS[@]}"

if [[ ${APPLY} -eq 0 ]]; then
    echo
    echo "Dry run only. Re-run with:"
    echo "  bash scripts/delete_seed_trans_dirs.sh --yes ${ROOT}"
    exit 0
fi

for dir in "${TARGETS[@]}"; do
    base="$(basename "${dir}")"
    parent="$(basename "$(dirname "${dir}")")"
    grandparent="$(basename "$(dirname "$(dirname "${dir}")")")"

    if [[ "${base}" != trans* ]]; then
        echo "Skipping unexpected match: ${dir}" >&2
        continue
    fi

    if [[ "${parent}" =~ ^seed[0-9]+$ ]]; then
        :
    elif [[ "${parent}" == "methods" && "${grandparent}" == "pipeline_val_new" ]]; then
        :
    else
        echo "Skipping unexpected match: ${dir}" >&2
        continue
    fi

    rm -rf -- "${dir}"
    echo "Deleted: ${dir}"
done
