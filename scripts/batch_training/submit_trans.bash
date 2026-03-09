#!/bin/bash

# Run trans experiments on three LIBERO benchmarks:
# - pi0_libero (pizero)
# - pi0_fast_libero (pizero_fast)
# - openvla_libero (openvla_libero_10)
#
# Usage:
#   bash scripts/batch_training/submit_trans.bash           # run all
#   bash scripts/batch_training/submit_trans.bash pi0       # only pi0_libero
#   bash scripts/batch_training/submit_trans.bash pi0fast   # only pi0_fast_libero
#   bash scripts/batch_training/submit_trans.bash openvla   # only openvla_libero
#   bash scripts/batch_training/submit_trans.bash all 0     # run all on GPU 0

set -euo pipefail

TARGET="${1:-all}"
GPU_ID="${2:-${CUDA_VISIBLE_DEVICES:-1}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SAFE_OPENPI_ROLLOUT_ROOT=/data1/mtr/data/safe_rollouts/
SAFE_OPENVLA_ROLLOUT_ROOT=/data1/mtr/data/safe_rollouts/openvla/
WANDB_DIR=./wandb_trans
SEEDS=0-1-2

run_pi0() {
  # Moderate sweep for pi0_libero:
  # - focus on feature extraction indices + mild regularization
  python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=pi0diff_libero_v1 \
    train.wandb_dir="${WANDB_DIR}/pi0_libero" \
    train.roc_every=10 \
    dataset=pizero \
    dataset.data_path_prefix="${SAFE_OPENPI_ROLLOUT_ROOT}" \
    dataset.horizon_idx_rel=mean \
    dataset.diff_idx_rel=mean \
    model=trans \
    model.lr=1e-4 \
    model.dropout=0.2 \
    model.lambda_reg=0.05 \
    model.use_time_weighting=True \
    model.lambda_pairwise_auc=0.01 \
    model.pairwise_auc_beta=5.0 \
    model.use_prefix_pairwise_auc=False \
    model.lambda_prefix_pairwise_auc=0.01 \
    model.prefix_pairwise_ratio=0.4 \
    train.seed="${SEEDS}" \
    train.exp_suffix=trans
}

run_pi0fast() {
  # Keep the current best-known pi0fast settings fixed.
  python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=pi0fast_libero_v4 \
    train.wandb_dir="${WANDB_DIR}/pi0fast_libero" \
    train.roc_every=10 \
    dataset=pizero_fast \
    dataset.data_path_prefix="${SAFE_OPENPI_ROLLOUT_ROOT}" \
    dataset.feat_name=encoded \
    dataset.token_idx_rel=mean \
    model=trans \
    model.lr=1e-4 \
    model.dropout=0.2 \
    model.lambda_reg=0.1 \
    model.use_time_weighting=True \
    model.lambda_pairwise_auc=0.0,0.1 \
    model.pairwise_auc_beta=5.0 \
    model.use_prefix_pairwise_auc=False,True \
    model.lambda_prefix_pairwise_auc=0.01,0.1 \
    model.prefix_pairwise_ratio=0.4 \
    train.seed="${SEEDS}" \
    train.exp_suffix=trans
}

run_openvla() {
  # Overfitting-focused sweep for openvla_libero.
  python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=openvla_libero_v2 \
    train.wandb_dir="${WANDB_DIR}/openvla" \
    train.roc_every=10 \
    dataset=openvla_libero_10 \
    dataset.data_path_prefix="${SAFE_OPENVLA_ROLLOUT_ROOT}" \
    dataset.token_idx_rel=mean,concat-2 \
    dataset.load_to_cuda=False \
    model=trans \
    model.lr=1e-5,3e-5 \
    model.dropout=0.3 \
    model.lambda_reg=0.1,0.3 \
    model.use_time_weighting=True \
    model.lambda_pairwise_auc=0.01 \
    model.pairwise_auc_beta=5.0 \
    model.use_prefix_pairwise_auc=False \
    model.lambda_prefix_pairwise_auc=0.01 \
    model.prefix_pairwise_ratio=0.4 \
    train.seed=0 \
    train.exp_suffix=trans
}

case "${TARGET}" in
  all)
    run_pi0
    run_pi0fast
    run_openvla
    ;;
  pi0)
    run_pi0
    ;;
  pi0fast)
    run_pi0fast
    ;;
  openvla)
    run_openvla
    ;;
  *)
    echo "Unknown target: ${TARGET}. Use all|pi0|pi0fast|openvla."
    exit 1
    ;;
esac
