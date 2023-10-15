#!/bin/bash

DATASET="MHCD"
MODEL_DIR="./models/v_info"
LOG_DIR="./logs/v_info"
BATCH_SIZE=1
LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-2

CUDA_VISIBLE_DEVICES=0 nohup python3 -u src/run_glue_no_trainer.py \
--label depression_label \
--model_name_or_path AIMH/mental-roberta-large \
--tokenizer_name AIMH/mental-roberta-large \
--train_file ./data/dataset_analysis/v_info/${DATASET}_std.csv \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} \
--num_train_epochs 2 \
--seed 0 \
--output_dir ${MODEL_DIR}/mental-roberta-large-${DATASET}-std > ${LOG_DIR}/mental-roberta-large-${DATASET}-std.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/run_glue_no_trainer.py \
# --label depression_label \
# --model_name_or_path AIMH/mental-roberta-large \
# --tokenizer_name AIMH/mental-roberta-large \
# --train_file ./data/dataset_analysis/v_info/${DATASET}_null.csv \
# --per_device_train_batch_size ${BATCH_SIZE} \
# --per_device_eval_batch_size ${BATCH_SIZE} \
# --learning_rate ${LEARNING_RATE} \
# --num_train_epochs 1 \
# --seed 0 \
# --output_dir ${MODEL_DIR}/mental-roberta-large-${DATASET}-null > ${LOG_DIR}/mental-roberta-large-${DATASET}-null.log &