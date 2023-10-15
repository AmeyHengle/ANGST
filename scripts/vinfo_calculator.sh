#!/bin/bash

DATASET="civilcomments"
DATA_DIR="dataset/v_info"
MODEL_DIR="models/v_info"
LOG_DIR="logs/v_info"

CUDA_VISIBLE_DEVICES=2 nohup python3 -u src/v_info.py \
--transform_type null \
--dataset ${DATASET} \
--dataset_dir ${DATA_DIR} \
--model_dir ${MODEL_DIR} > ${LOG_DIR}/v_info_${DATASET}_null.log &

CUDA_VISIBLE_DEVICES=2 nohup python3 -u src/v_info.py \
--transform_type bad_vocab \
--dataset ${DATASET} \
--dataset_dir ${DATA_DIR} \
--model_dir ${MODEL_DIR} > ${LOG_DIR}/v_info_${DATASET}_bad_vocab.log &

CUDA_VISIBLE_DEVICES=3 nohup python3 -u src/v_info.py \
--transform_type sentiment_vocab \
--dataset ${DATASET} \
--dataset_dir ${DATA_DIR} \
--model_dir ${MODEL_DIR} > ${LOG_DIR}/v_info_${DATASET}_sentiment_vocab.log &

CUDA_VISIBLE_DEVICES=3 nohup python3 -u src/v_info.py \
--transform_type sentiment \
--dataset ${DATASET} \
--dataset_dir ${DATA_DIR} \
--model_dir ${MODEL_DIR} > ${LOG_DIR}/v_info_${DATASET}_sentiment.log &