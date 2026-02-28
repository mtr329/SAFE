#!/bin/bash

# Run all experiments for Pi0-FAST model on the LIBERO rollouts

GROUP_NAME=pi0fast_libero_v4
export CUDA_VISIBLE_DEVICES=2
SAFE_OPENPI_ROLLOUT_ROOT=/data1/mtr/data/safe_rollouts/
WANDB_DIR=./wandb_trans

python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    train.wandb_dir=${WANDB_DIR} \
    train.roc_every=10 \
    dataset=pizero_fast \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    dataset.feat_name=encoded \
    dataset.token_idx_rel=mean \
    model=trans \
    model.lambda_pairwise_auc=0.01,0.1 \
    model.pairwise_auc_beta=5.0,10.0 \
    model.dropout=0.0,0.1 \
    model.lr=1e-4,5e-4 \
    model.lambda_reg=0.05,0.1 \
    model.use_time_weighting=False,True \
    train.seed=0 \
    train.exp_suffix=trans
