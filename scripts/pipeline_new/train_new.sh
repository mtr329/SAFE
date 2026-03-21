#!/bin/bash

set -euo pipefail

# Thin wrapper around failure_prob/pipeline/train_new.py.
#
# Usage:
#   scripts/pipeline_new/train_new.sh dataset=pizero_fast model=lstm train.exp_suffix=lstm
#   CUDA_VISIBLE_DEVICES=3 scripts/pipeline_new/train_new.sh --multirun ...

cmd=(
    python -m failure_prob.pipeline.train_new
    "$@"
)

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

env PYTHONPATH=. "${cmd[@]}"
