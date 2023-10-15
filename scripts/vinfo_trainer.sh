#!/bin/bash

DATASET="civilcomments"
MODEL_DIR="models/v_info"
LOG_DIR="logs/v_info"
BATCH_SIZE=32
LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-2

# CUDA_VISIBLE_DEVICES=0 nohup python3 -u src/run_glue_no_trainer.py \
# --model_name_or_path bert-base-cased \
# --tokenizer_name bert-base-cased \
# --train_file dataset/v_info/${DATASET}/${DATASET}_std.csv \
# --validation_file dataset/v_info/${DATASET}/${DATASET}_std.csv \
# --per_device_train_batch_size ${BATCH_SIZE} \
# --per_device_eval_batch_size ${BATCH_SIZE} \
# --learning_rate ${LEARNING_RATE} \
# --num_train_epochs 2 \
# --seed 0 \
# --output_dir ${MODEL_DIR}/finetuned/bert-base-cased-${DATASET}-std > ${LOG_DIR}/bert-base-cased-${DATASET}-std.log &

# CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/run_glue_no_trainer.py \
# --model_name_or_path bert-base-cased \
# --tokenizer_name bert-base-cased \
# --train_file dataset/v_info/${DATASET}/${DATASET}_null.csv \
# --validation_file dataset/v_info/${DATASET}/${DATASET}_std.csv \
# --per_device_train_batch_size ${BATCH_SIZE} \
# --per_device_eval_batch_size ${BATCH_SIZE} \
# --learning_rate ${LEARNING_RATE} \
# --num_train_epochs 1 \
# --seed 0 \
# --output_dir ${MODEL_DIR}/finetuned/bert-base-cased-${DATASET}-null > ${LOG_DIR}/bert-base-cased-${DATASET}-null.log &

CUDA_VISIBLE_DEVICES=3 nohup python3 -u src/run_glue_no_trainer.py \
--model_name_or_path bert-base-cased \
--tokenizer_name bert-base-cased \
--train_file dataset/v_info/${DATASET}/${DATASET}_sentiment.csv \
--validation_file dataset/v_info/${DATASET}/${DATASET}_std.csv \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} \
--num_train_epochs 1 \
--seed 0 \
--output_dir ${MODEL_DIR}/finetuned/bert-base-cased-${DATASET}-sentiment > ${LOG_DIR}/bert-base-cased-${DATASET}-sentiment.log &

CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/run_glue_no_trainer.py \
--model_name_or_path bert-base-cased \
--tokenizer_name bert-base-cased \
--train_file dataset/v_info/${DATASET}/${DATASET}_sentiment_vocab.csv \
--validation_file dataset/v_info/${DATASET}/${DATASET}_std.csv \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} \
--num_train_epochs 1 \
--seed 0 \
--output_dir ${MODEL_DIR}/finetuned/bert-base-cased-${DATASET}-sentiment-vocab > ${LOG_DIR}/bert-base-cased-${DATASET}-sentiment-vocab.log &

CUDA_VISIBLE_DEVICES=2 nohup python3 -u src/run_glue_no_trainer.py \
--model_name_or_path bert-base-cased \
--tokenizer_name bert-base-cased \
--train_file dataset/v_info/${DATASET}/${DATASET}_bad_vocab.csv \
--validation_file dataset/v_info/${DATASET}/${DATASET}_std.csv \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size ${BATCH_SIZE} \
--learning_rate ${LEARNING_RATE} \
--num_train_epochs 1 \
--seed 0 \
--output_dir ${MODEL_DIR}/finetuned/bert-base-cased-${DATASET}-bad-vocab > ${LOG_DIR}/bert-base-cased-${DATASET}-bad-vocab.log &

