#!/bin/bash

# DATASET="MHCD"
# DATASET="DATD"
# DATASET="dreddit"
DATASET="dep_reddit"
# DATASET="SDCNL"

DATA_DIR=data/dataset_analysis/v_info/
MODEL_DIR="models/v_info"
LOG_DIR="logs/v_info"

CUDA_VISIBLE_DEVICES=3 nohup python3 -u src/v_info.py \
--dataset ${DATASET} \
--model roberta-base \
--dataset_dir ${DATA_DIR} \
--model_dir ${MODEL_DIR} > ${LOG_DIR}/v_info_roberta-base_${DATASET}_null.log &