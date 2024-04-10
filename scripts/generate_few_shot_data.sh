#!/bin/bash

DATA_PATH="./data/test/full_test.csv"
RESULT_DIR="./results/zero_shot"
VERSION=3

MODEL="mental_llama_chat_7b"
# DATA_TYPE="depression"
# DATA_TYPE="anxiety"
DATA_TYPE="comorbidity"
OUTFILE=./data/few_shot_prompts/${DATA_TYPE}_llama_num_examples_ss_2.csv

python3 src/generate_few_shot_data.py \
--data_type $DATA_TYPE \
--use_semantic_similarity_only \
--num_icl_examples 2 \
--model $MODEL \
--outfile $OUTFILE