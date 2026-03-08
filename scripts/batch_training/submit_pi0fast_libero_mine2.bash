#!/bin/bash

# Run all experiments for Pi0-FAST model on the LIBERO rollouts

GROUP_NAME=pi0fast_libero_v4
export CUDA_VISIBLE_DEVICES=1
SAFE_OPENPI_ROLLOUT_ROOT=/data1/mtr/data/safe_rollouts/
WANDB_DIR=./wandb_trans

# Early val_unseen focused sweep:
# 1) keep stable lr
# 2) strengthen regularization
# 3) reduce objective bias toward end-of-trajectory max scores
# 4) add early-prefix pairwise objective
python -m failure_prob.train \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    train.wandb_dir=${WANDB_DIR} \
    train.roc_every=10 \
    dataset=pizero_fast \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    dataset.feat_name=encoded,pre_logits \
    dataset.token_idx_rel=mean \
    model=trans \
    model.lambda_pairwise_auc=0.0,0.01 \
    model.pairwise_auc_beta=5.0 \
    model.use_prefix_pairwise_auc=True \
    model.lambda_prefix_pairwise_auc=0.01,0.05 \
    model.prefix_pairwise_ratio=0.3,0.4 \
    model.dropout=0.2,0.3,0.5 \
    model.lr=1e-4 \
    model.lambda_reg=0.1,0.2 \
    model.use_time_weighting=True \
    train.seed=0 \
    train.exp_suffix=trans
