#!/bin/bash

# Run all experiments for Pi0-FAST model on the LIBERO rollouts

GROUP_NAME=pi0fast_libero_v4
export CUDA_VISIBLE_DEVICES=3
SAFE_OPENPI_ROLLOUT_ROOT=/data1/mtr/data/safe_rollouts/

# LSTM
# python -m failure_prob.train \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.feat_name=encoded \
#     dataset.token_idx_rel=mean \
#     model=lstm \
#     model.lr=3.0e-4 \
#     model.lambda_reg=1.0e-3 \
#     train.seed=0-1-2 \
#     train.exp_suffix=lstm

# # MLP
# python -m failure_prob.train \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.feat_name=pre_logits \
#     dataset.token_idx_rel=1.0 \
#     model=indep \
#     model.lr=1.0e-4 \
#     model.lambda_reg=1.0e-2 \
#     train.seed=0-1-2 \
#     train.exp_suffix=mlp

# The embed baseline
# python -m failure_prob.train \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.feat_name=pre_logits \
#     dataset.token_idx_rel=mean \
#     model=embed \
#     model.n_epochs=1 \
#     model.distance=cosine,euclid \
#     model.use_success_only=False \
#     model.topk=10 \
#     model.cumsum=False \
#     train.seed=0-1-2 \
#     train.exp_suffix=embed

# python -m failure_prob.train \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.feat_name=pre_logits \
#     dataset.token_idx_rel=mean \
#     model=embed \
#     model.n_epochs=1 \
#     model.distance=mahala \
#     model.use_success_only=False \
#     model.cumsum=False \
#     train.seed=0-1-2 \
#     train.exp_suffix=embed

# python -m failure_prob.train \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     dataset.feat_name=encoded \
#     dataset.token_idx_rel=0.0 \
#     model=embed \
#     model.distance=pca_kmeans \
#     model.pca_dim=32 \
#     model.n_clusters=16 \
#     model.use_success_only=False \
#     model.cumsum=True \
#     train.seed=0-1-2 \
#     train.exp_suffix=embed

# Chen's method
# logpzo: cuda oom
# for MODEL in rnd logpzo; do
# for MODEL in logpzo; do
# for FEAT in pre_logits; do
#     python -m failure_prob.train \
#         --multirun \
#         train.wandb_group_name=${GROUP_NAME} \
#         dataset=pizero_fast \
#         dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#         dataset.feat_name=${FEAT} \
#         dataset.token_idx_rel=mean \
#         model=${MODEL} \
#         model.use_success_only=False \
#         model.batch_size=32 \
#         train.roc_every=50 \
#         train.seed=0-1-2 \
#         train.exp_suffix=chen
# done
# done

# # The hand-crafted baselines
# error
# python -m failure_prob.train \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     dataset=pizero_fast \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     train.log_precomputed_only=True \
#     train.seed=0-1-2 \
#     train.exp_suffix=handcrafted


# # Handcreafted baselines with multiple action samples
# python -m failure_prob.train \
#     --multirun \
#     train.wandb_group_name=${GROUP_NAME} \
#     dataset=pizero_fast_libero_sample10 \
#     dataset.data_path_prefix=${SAFE_OPENPI_ROLLOUT_ROOT} \
#     train.log_precomputed_only=True \
#     train.seed=0-1-2 \
#     train.exp_suffix=handcrafted_multi

