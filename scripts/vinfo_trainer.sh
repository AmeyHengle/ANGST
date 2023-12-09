#!/bin/bash

DATASET="MHCD"
<<<<<<< Updated upstream
# DATASET="DATD"
# DATASET="dreddit"
# DATASET="dep_reddit"
# DATASET="SDCNL"

DATA_DIR=data/dataset_analysis/v_info/
MODEL_DIR="models/v_info"
LOG_DIR="logs/v_info"

BATCH_SIZE=16
LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-2

# source activate llm_env

CUDA_VISIBLE_DEVICES=2 nohup python3 -u src/vinfo_trainer.py \
--label anxiety_label \
--model_name_or_path roberta-base \
--tokenizer_name roberta-base \
--train_file ${DATA_DIR}/${DATASET}_std.csv \
=======
MODEL_DIR="models/v_info"
LOG_DIR="logs/v_info"
BATCH_SIZE=8
LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-2

source activate llm_env

CUDA_VISIBLE_DEVICES=0 nohup python3 -u src/run_glue_no_trainer.py \
--label depression_label \
--model_name_or_path AIMH/mental-roberta-large \
--tokenizer_name AIMH/mental-roberta-large \
--train_file data/dataset_analysis/v_info/${DATASET}_std.csv \
>>>>>>> Stashed changes
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} \
--num_train_epochs 1 \
--seed 0 \
<<<<<<< Updated upstream
--output_dir ${MODEL_DIR}/roberta-base-${DATASET}-std-anxiety_label > ${LOG_DIR}/roberta-base-${DATASET}-std-anxiety_label.log &

CUDA_VISIBLE_DEVICES=3 nohup python3 -u src/vinfo_trainer.py \
--label anxiety_label \
--model_name_or_path roberta-base \
--tokenizer_name roberta-base \
--train_file ${DATA_DIR}/${DATASET}_null.csv \
=======
--output_dir ${MODEL_DIR}/mental-roberta-large-${DATASET}-std-depression_label > ${LOG_DIR}/mental-roberta-large-${DATASET}-std-depression_label.log &

CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/run_glue_no_trainer.py \
--label depression_label \
--model_name_or_path AIMH/mental-roberta-large \
--tokenizer_name AIMH/mental-roberta-large \
--train_file data/dataset_analysis/v_info/${DATASET}_null.csv \
>>>>>>> Stashed changes
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} \
--num_train_epochs 1 \
--seed 0 \
<<<<<<< Updated upstream
--output_dir ${MODEL_DIR}/roberta-base-${DATASET}-null-anxiety_label > ${LOG_DIR}/roberta-base-${DATASET}-null-anxiety_label.log &
=======
--output_dir ${MODEL_DIR}/mental-roberta-large-${DATASET}-null-depression_label > ${LOG_DIR}/mental-roberta-large-${DATASET}-null-depression_label.log &
>>>>>>> Stashed changes
