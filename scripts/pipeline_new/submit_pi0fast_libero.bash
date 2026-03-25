#!/bin/bash

set -euo pipefail

# Run all experiments for Pi0-FAST model on the LIBERO rollouts.

GROUP_NAME=pi0fast_libero_v4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
SAFE_OPENPI_ROLLOUT_ROOT=/data1/mtr/data/safe_rollouts/
WANDB_DIR=./wandb_new
CACHE_DIR=./dataset_cache

# # LSTM
# python -m failure_prob.pipeline.train_new \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     train.wandb_dir=${WANDB_DIR}/pizero_fast/lstm \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.use_cache=True \
#     dataset.cache_dir=${CACHE_DIR} \
#     dataset.feat_name=encoded,pre_logits \
#     dataset.token_idx_rel=0.0,1.0,mean \
#     model=lstm \
#     model.lr=3e-5,1e-4,3e-4,1e-3 \
#     model.lambda_reg=1e-3,1e-2,1e-1 \
#     train.seed=0-1-2 \
#     train.exp_suffix=lstm

# # MLP
# python -m failure_prob.pipeline.train_new \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     train.wandb_dir=${WANDB_DIR}/pizero_fast/indep \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.use_cache=True \
#     dataset.cache_dir=${CACHE_DIR} \
#     dataset.feat_name=encoded,pre_logits \
#     dataset.token_idx_rel=0.0,1.0,mean \
#     model=indep \
#     model.lr=1e-5,1e-4,3e-4,1e-3 \
#     model.lambda_reg=1e-3,1e-2,1e-1 \
#     train.seed=0-1-2 \
#     train.exp_suffix=mlp

# # The embed baseline
# python -m failure_prob.pipeline.train_new \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     train.wandb_dir=${WANDB_DIR}/pizero_fast/embed_\${model.distance} \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.use_cache=True \
#     dataset.cache_dir=${CACHE_DIR} \
#     dataset.feat_name=encoded,pre_logits \
#     dataset.token_idx_rel=0.0,1.0,mean \
#     model=embed \
#     model.n_epochs=1 \
#     model.distance=cosine,euclid \
#     model.use_success_only=False \
#     model.topk=1,5,10 \
#     model.cumsum=False,True \
#     train.seed=0-1-2 \
#     train.exp_suffix=embed

# python -m failure_prob.pipeline.train_new \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     train.wandb_dir=${WANDB_DIR}/pizero_fast/embed_\${model.distance} \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.use_cache=True \
#     dataset.cache_dir=${CACHE_DIR} \
#     dataset.feat_name=encoded,pre_logits \
#     dataset.token_idx_rel=0.0,1.0,mean \
#     model=embed \
#     model.n_epochs=1 \
#     model.distance=mahala \
#     model.use_success_only=False \
#     model.cumsum=False,True \
#     train.seed=0-1-2 \
#     train.exp_suffix=embed

# python -m failure_prob.pipeline.train_new \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     train.wandb_dir=${WANDB_DIR}/pizero_fast/embed_\${model.distance} \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.use_cache=True \
#     dataset.cache_dir=${CACHE_DIR} \
#     dataset.feat_name=encoded,pre_logits \
#     dataset.token_idx_rel=0.0,1.0,mean \
#     model=embed \
#     model.distance=pca_kmeans \
#     model.pca_dim=32,64,128 \
#     model.n_clusters=16,32,64 \
#     model.use_success_only=False \
#     model.cumsum=False,True \
#     train.seed=0-1-2 \
#     train.exp_suffix=embed

# # Chen's method
# # logpzo: cuda oom, use batch_size=24
# for MODEL in rnd; do
# for FEAT in encoded pre_logits; do
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/pizero_fast/${MODEL} \
#         dataset=pizero_fast \
#         dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.feat_name=${FEAT} \
#         dataset.token_idx_rel=0.0,1.0,mean \
#         model=${MODEL} \
#         model.use_success_only=False \
#         model.batch_size=32 \
#         train.roc_every=50 \
#         train.seed=0-1-2 \
#         train.exp_suffix=chen
# done
# done

# for MODEL in logpzo; do
# for FEAT in pre_logits; do
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/pizero_fast/${MODEL} \
#         dataset=pizero_fast \
#         dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.feat_name=${FEAT} \
#         dataset.token_idx_rel=0.0,1.0,mean \
#         model=${MODEL} \
#         model.use_success_only=False \
#         model.batch_size=24 \
#         train.roc_every=50 \
#         train.seed=0-1-2 \
#         train.exp_suffix=chen
# done
# done

# # The hand-crafted baselines
# python -m failure_prob.pipeline.train_new \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     train.wandb_dir=${WANDB_DIR}/pizero_fast/handcrafted \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.use_cache=True \
#     dataset.cache_dir=${CACHE_DIR} \
#     train.log_precomputed_only=True \
#     train.seed=0-1-2 \
#     train.exp_suffix=handcrafted

# Trans
# Small shared tradeoff sweep.
# Use the same model search space across all three datasets and keep the budget
# small by only scanning three knobs that most directly affect pareto vs AUC.
# Total: 2 history sizes x 2 prefix-pairwise weights x 2 soft-detection weights
# x 3 seeds = 24 runs.
python -m failure_prob.pipeline.train_new \
    --multirun \
    train.wandb_group_name=${GROUP_NAME} \
    train.wandb_dir=${WANDB_DIR}/pizero_fast/trans \
    dataset=pizero_fast \
    dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
    dataset.use_cache=True \
    dataset.cache_dir=${CACHE_DIR} \
    dataset.feat_name=pre_logits \
    dataset.token_idx_rel=mean \
    model=trans \
    train.roc_every=10 \
    model.optimizer=adamw \
    model.lr=1e-4 \
    model.weight_decay=1e-4 \
    model.warmup_steps=500 \
    model.hidden_dim=128 \
    model.ff_dim=256 \
    model.n_layers=2 \
    model.n_heads=4 \
    model.dropout=0.15 \
    model.lambda_reg=0.05 \
    model.cumsum=True \
    model.use_time_weighting=True \
    model.use_class_conditional_time_weights=True \
    model.n_history_steps=16,24 \
    model.aux_warmup_epochs=10 \
    model.aux_ramp_epochs=25 \
    model.lambda_pairwise_auc=0.03 \
    model.pairwise_auc_beta=5.0 \
    model.use_prefix_pairwise_auc=True \
    model.lambda_prefix_pairwise_auc=0.04,0.08 \
    model.prefix_pairwise_ratios='[0.1,0.2,0.35]' \
    model.prefix_pairwise_weights='[0.5,1.0,0.7]' \
    model.prefix_pairwise_time_discount_gamma=0.25 \
    model.lambda_prefix_monitor=0.05 \
    model.prefix_monitor_ratios='[0.1,0.2,0.35]' \
    model.prefix_monitor_weights='[0.5,1.0,0.7]' \
    model.use_soft_detection_loss=True \
    model.lambda_soft_detection=0.05,0.10 \
    model.soft_detection_threshold=0.45 \
    model.soft_detection_temperature=0.08 \
    train.seed=0-1-2 \
    train.exp_suffix=trans_shared_small_scan
