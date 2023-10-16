#!/bin/bash

DATASET="MHCD"
# DATASET="DATD"
# DATASET="dreddit"
# DATASET="dep_reddit"
# DATASET="SDCNL"

DATA_DIR=data/dataset_analysis/v_info/
MODEL_DIR=models/v_info
LOG_DIR=logs/v_info

CUDA_VISIBLE_DEVICES=0 nohup python3 -u src/v_info.py \
--label depression_label \
--transform_type null \
--dataset ${DATASET} \
--dataset_dir ${DATA_DIR} \
--model_dir ${MODEL_DIR} > ${LOG_DIR}/v_info_${DATASET}_null.log &