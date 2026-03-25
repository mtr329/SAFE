#!/bin/bash

set -euo pipefail

# Run all experiments for Pi0 model on the LIBERO rollouts.

GROUP_NAME=pi0diff_libero_v1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export WANDB_MODE="${WANDB_MODE:-offline}"
SAFE_OPENPI_ROLLOUT_ROOT=/data1/mtr/data/safe_rollouts/
WANDB_DIR=./wandb_new
CACHE_DIR=./dataset_cache

# # LSTM and MLP
# for REG in 1e-3 1e-2 1e-1; do
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/pizero/lstm \
#         dataset=pizero \
#         dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.horizon_idx_rel=0.0,1.0,concat-2 \
#         dataset.diff_idx_rel=0.0,1.0,concat-2 \
#         model=lstm \
#         model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
#         model.lambda_reg=${REG} \
#         train.seed=0-1-2 \
#         train.exp_suffix=lstm
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/pizero/indep \
#         dataset=pizero \
#         dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.horizon_idx_rel=0.0,1.0,concat-2 \
#         dataset.diff_idx_rel=0.0,1.0,concat-2 \
#         model=indep \
#         model.lr=1e-5,3e-5,1e-4,3e-4,1e-3 \
#         model.lambda_reg=${REG} \
#         train.seed=0-1-2 \
#         train.exp_suffix=mlp
# done

# # Embed model
# for DIST in cosine euclid; do
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/pizero/embed_\${model.distance} \
#         dataset=pizero \
#         dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.horizon_idx_rel=0.0,1.0,concat-2 \
#         dataset.diff_idx_rel=0.0,1.0,concat-2 \
#         dataset.load_to_cuda=False \
#         model=embed \
#         model.n_epochs=1 \
#         model.distance=${DIST} \
#         model.use_success_only=False \
#         model.topk=1,5,10 \
#         model.cumsum=False,True \
#         train.seed=0-1-2 \
#         train.exp_suffix=embed
# done

# python -m failure_prob.pipeline.train_new \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     train.wandb_dir=${WANDB_DIR}/pizero/embed_\${model.distance} \
#     dataset=pizero \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.use_cache=True \
#     dataset.cache_dir=${CACHE_DIR} \
#     dataset.horizon_idx_rel=0.0,1.0,concat-2 \
#     dataset.diff_idx_rel=0.0,1.0,concat-2 \
#     dataset.load_to_cuda=False \
#     model=embed \
#     model.n_epochs=1 \
#     model.distance=mahala \
#     model.use_success_only=False \
#     model.cumsum=False,True \
#     train.seed=0-1-2 \
#     train.exp_suffix=embed

# for PCA_DIM in 32 64 128; do
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/pizero/embed_\${model.distance} \
#         dataset=pizero \
#         dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.horizon_idx_rel=0.0,1.0,concat-2 \
#         dataset.diff_idx_rel=0.0,1.0,concat-2 \
#         dataset.load_to_cuda=False \
#         model=embed \
#         model.distance=pca_kmeans \
#         model.pca_dim=${PCA_DIM} \
#         model.n_clusters=16,32,64 \
#         model.use_success_only=False \
#         model.cumsum=False,True \
#         train.seed=0-1-2 \
#         train.exp_suffix=embed
# done

# # Chen's baselines
# for HORIZON_IDX in 0.0 1.0 concat-2; do
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/pizero/rnd \
#         dataset=pizero \
#         dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.horizon_idx_rel=${HORIZON_IDX} \
#         dataset.diff_idx_rel=0.0,1.0,concat-2 \
#         dataset.load_to_cuda=False \
#         model=rnd \
#         train.roc_every=50 \
#         model.batch_size=32 \
#         model.use_success_only=False \
#         train.seed=0-1-2 \
#         train.exp_suffix=chen
#     python -m failure_prob.pipeline.train_new \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         train.wandb_dir=${WANDB_DIR}/pizero/logpzo \
#         dataset=pizero \
#         dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#         dataset.use_cache=True \
#         dataset.cache_dir=${CACHE_DIR} \
#         dataset.horizon_idx_rel=${HORIZON_IDX} \
#         dataset.diff_idx_rel=0.0,1.0,concat-2 \
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
# python -m failure_prob.pipeline.train_new \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     train.wandb_dir=${WANDB_DIR}/pizero/handcrafted \
#     dataset=pizero \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.use_cache=True \
#     dataset.cache_dir=${CACHE_DIR} \
#     train.log_precomputed_only=True \
#     train.seed=0-1-2 \
#     train.exp_suffix=handcrafted

# Trans
# Small shared local sweep around the current best trans settings.
# Only explore new points that were not in the previous scan:
#   - keep the strong shared defaults fixed,
#   - drop the weaker 32-step history,
#   - turn on time gating,
#   - probe two new soft-detection weights and two gate inits.
# Total: 2 history sizes x 2 soft-detection weights x 2 gate inits x 3 seeds = 24 runs.
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
    model.n_history_steps=16,24 \
    model.aux_warmup_epochs=10 \
    model.aux_ramp_epochs=25 \
    model.lambda_pairwise_auc=0.03 \
    model.pairwise_auc_beta=5.0 \
    model.use_time_gate=True \
    model.time_gate_tau_init=0.15,0.30 \
    model.use_prefix_pairwise_auc=True \
    model.lambda_prefix_pairwise_auc=0.04 \
    model.prefix_pairwise_ratios='[0.1,0.2,0.35]' \
    model.prefix_pairwise_weights='[0.5,1.0,0.7]' \
    model.prefix_pairwise_time_discount_gamma=0.25 \
    model.lambda_prefix_monitor=0.05 \
    model.prefix_monitor_ratios='[0.1,0.2,0.35]' \
    model.prefix_monitor_weights='[0.5,1.0,0.7]' \
    model.use_soft_detection_loss=True \
    model.lambda_soft_detection=0.03,0.08 \
    model.soft_detection_threshold=0.45 \
    model.soft_detection_temperature=0.08 \
    train.seed=0-1-2 \
    train.exp_suffix=trans_shared_local_gate_scan
