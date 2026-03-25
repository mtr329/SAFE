#!/bin/bash

set -euo pipefail

# Run all experiments for OpenVLA model on the LIBERO rollouts.

GROUP_NAME=openvla_libero_v2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export WANDB_MODE="${WANDB_MODE:-offline}"
SAFE_OPENVLA_ROLLOUT_ROOT=/data1/mtr/data/safe_rollouts/openvla/
WANDB_DIR=./wandb_new
CACHE_DIR=./dataset_cache
OPENVLA_USE_CACHE="${OPENVLA_USE_CACHE:-True}"
OPENVLA_REFRESH_CACHE="${OPENVLA_REFRESH_CACHE:-False}"
OPENVLA_TRANS_BATCH_SIZE="${OPENVLA_TRANS_BATCH_SIZE:-32}"

# # LSTM and MLP
# for SUITE_NAME in 10; do
# for SEED in 0 1 2; do
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/openvla/lstm \
#         dataset=openvla_libero_${SUITE_NAME} \
#         dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.token_idx_rel=mean,0.0,1.0 \
#         dataset.load_to_cuda=False \
#         model=lstm \
#         model.batch_size=64 \
#         model.lr=1e-4,3e-4,1e-3 \
#         model.lambda_reg=1e-3,1e-2,1e-1,1 \
#         train.seed=${SEED} \
#         train.exp_suffix=lstm
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/openvla/indep \
#         dataset=openvla_libero_${SUITE_NAME} \
#         dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.token_idx_rel=mean,0.0,1.0 \
#         dataset.load_to_cuda=False \
#         model=indep \
#         model.batch_size=64 \
#         model.lr=1e-4,3e-4,1e-3 \
#         model.lambda_reg=1e-3,1e-2,1e-1,1 \
#         train.seed=${SEED} \
#         train.exp_suffix=mlp
# done
# done

# # Embedding-based method
# for SUITE_NAME in 10; do
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/openvla/embed_\${model.distance} \
#         dataset=openvla_libero_${SUITE_NAME} \
#         dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.token_idx_rel=mean,0.0,1.0 \
#         dataset.load_to_cuda=False \
#         model=embed \
#         model.n_epochs=1 \
#         model.distance=cosine,euclid \
#         model.use_success_only=False \
#         model.topk=1,5,10 \
#         model.cumsum=False,True \
#         train.seed=0-1-2 \
#         train.exp_suffix=embed
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/openvla/embed_\${model.distance} \
#         dataset=openvla_libero_${SUITE_NAME} \
#         dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.token_idx_rel=mean,0.0,1.0 \
#         dataset.load_to_cuda=False \
#         model=embed \
#         model.n_epochs=1 \
#         model.distance=mahala \
#         model.use_success_only=False \
#         model.cumsum=False,True \
#         train.seed=0-1-2 \
#         train.exp_suffix=embed
# done

# for SUITE_NAME in 10; do
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/openvla/embed_\${model.distance} \
#         dataset=openvla_libero_${SUITE_NAME} \
#         dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.token_idx_rel=mean,0.0,1.0 \
#         dataset.load_to_cuda=False \
#         model=embed \
#         model.distance=pca_kmeans \
#         model.pca_dim=32,64,128 \
#         model.n_clusters=16,32,64 \
#         model.use_success_only=False \
#         model.cumsum=False,True \
#         train.seed=0-1-2 \
#         train.exp_suffix=embed
# done

# Chen's baselines
# for SUITE_NAME in 10; do
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/openvla/rnd \
#         dataset=openvla_libero_${SUITE_NAME} \
#         dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.token_idx_rel=mean,0.0,1.0 \
#         dataset.load_to_cuda=False \
#         model=rnd \
#         train.roc_every=50 \
#         model.batch_size=24 \
#         model.use_success_only=False \
#         train.seed=0-1-2 \
#         train.exp_suffix=chen
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/openvla/logpzo \
#         dataset=openvla_libero_${SUITE_NAME} \
#         dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.token_idx_rel=mean,0.0,1.0 \
#         dataset.load_to_cuda=False \
#         model=logpzo \
#         train.roc_every=50 \
#         model.batch_size=24 \
#         model.forward_chunk_size=512 \
#         model.use_success_only=False \
#         train.seed=0-1-2 \
#         train.exp_suffix=chen
# done

# # Handcrafted metrics
# for SUITE_NAME in 10; do
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/openvla/handcrafted \
#         dataset=openvla_libero_${SUITE_NAME} \
#         dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         train.log_precomputed_only=True \
#         train.seed=0-1-2 \
#         train.exp_suffix=handcrafted
# done

# Trans
# Small shared tradeoff sweep.
# Use the same model search space across all three datasets and keep the budget
# small by only scanning three knobs that most directly affect pareto vs AUC.
# Add one longer history option while keeping the rest fixed, but only scan one
# extra tradeoff knob to keep the budget small.
# Total: 3 history sizes x 2 soft-detection weights x 3 seeds = 18 runs per dataset.
for SUITE_NAME in 10; do
    python -m failure_prob.pipeline.train_new \
        --multirun \
        train.wandb_group_name=${GROUP_NAME} \
        train.wandb_dir=${WANDB_DIR}/openvla/trans \
        train.roc_every=100 \
        dataset=openvla_libero_${SUITE_NAME} \
        dataset.data_path_prefix=${SAFE_OPENVLA_ROLLOUT_ROOT} \
        dataset.use_cache=${OPENVLA_USE_CACHE} \
        dataset.refresh_cache=${OPENVLA_REFRESH_CACHE} \
        dataset.cache_dir=${CACHE_DIR} \
        dataset.token_idx_rel=mean \
        dataset.load_to_cuda=False \
        model=trans \
        model.batch_size=${OPENVLA_TRANS_BATCH_SIZE} \
        model.optimizer=adamw \
        model.lr=1e-4 \
        model.weight_decay=1e-4 \
        model.warmup_steps=40 \
        model.hidden_dim=128 \
        model.ff_dim=256 \
        model.n_layers=2 \
        model.n_heads=4 \
        model.dropout=0.15 \
        model.lambda_reg=0.05 \
        model.cumsum=True \
        model.use_time_weighting=True \
        model.use_class_conditional_time_weights=True \
        model.n_history_steps=16,24,32 \
        model.aux_warmup_epochs=10 \
        model.aux_ramp_epochs=25 \
        model.lambda_pairwise_auc=0.03 \
        model.pairwise_auc_beta=5.0 \
        model.use_prefix_pairwise_auc=True \
        model.lambda_prefix_pairwise_auc=0.04 \
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
done
