#!/bin/bash

# DATASET="MHCD"
# DATASET="DATD"
# DATASET="dreddit"
DATASET="dep_reddit"
# DATASET="SDCNL"

DATA_DIR=data/dataset_analysis/v_info/
MODEL_DIR="models/v_info"
LOG_DIR="logs/v_info"

BATCH_SIZE=16
LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-2

# source activate llm_env

CUDA_VISIBLE_DEVICES=0 nohup python3 -u src/vinfo_trainer.py \
--model_name_or_path AIMH/mental-bert-base-cased \
--tokenizer_name AIMH/mental-bert-base-cased \
--train_file ${DATA_DIR}/${DATASET}_std.csv \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} \
--num_train_epochs 1 \
--seed 0 \
--output_dir ${MODEL_DIR}/mental-bert-base-cased-${DATASET}-std > ${LOG_DIR}/mental-bert-base-cased-${DATASET}-std.log &

CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/vinfo_trainer.py \
--model_name_or_path AIMH/mental-bert-base-cased \
--tokenizer_name AIMH/mental-bert-base-cased \
--train_file ${DATA_DIR}/${DATASET}_null.csv \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} \
--num_train_epochs 1 \
--seed 0 \
--output_dir ${MODEL_DIR}/mental-bert-base-cased-${DATASET}-null > ${LOG_DIR}/mental-bert-base-cased-${DATASET}-null.log &
