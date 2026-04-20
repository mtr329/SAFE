#!/bin/bash

set -euo pipefail

# Run the Transformer failure model on Pi0 LIBERO rollouts.

GROUP_NAME=pi0diff_libero_v1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export WANDB_MODE="${WANDB_MODE:-offline}"
SAFE_OPENPI_ROLLOUT_ROOT=/data1/mtr/data/safe_rollouts/
WANDB_DIR=./wandb_new
CACHE_DIR=./dataset_cache

python -m failure_prob.pipeline.train_new \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    train.wandb_dir=${WANDB_DIR}/pizero/trans \
    train.roc_every=100 \
    dataset=pizero \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    dataset.use_cache=True \
    dataset.cache_dir=${CACHE_DIR} \
    dataset.horizon_idx_rel=mean \
    dataset.diff_idx_rel=mean \
    model=trans \
    model.optimizer=adamw \
    model.lr=1e-4,5e-4 \
    model.weight_decay=1e-4,5e-4 \
    model.warmup_steps=40 \
    model.hidden_dim=128 \
    model.ff_dim=256 \
    model.n_layers=2 \
    model.n_heads=4 \
    model.dropout=0.15 \
    model.lambda_reg=0.05 \
    model.cumsum=True \
    model.n_history_steps=16,24 \
    train.seed=0-1-2 \
    train.exp_suffix=trans
