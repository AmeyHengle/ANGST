#!/bin/bash

DATA_PATH="./data/test/full_test.csv"
RESULT_DIR="./results/zero_shot"

MODEL="llama_chat_7b"
DATA_TYPE="depression"
# DATA_TYPE="anxiety"
# DATA_TYPE="comorbidity"

NUM_EXAMPLES_PER_LABEL=2
OUTFILE=./data/few_shot_prompts/${DATA_TYPE}_num_examples_ss_${NUM_EXAMPLES_PER_LABEL}.csv

python3 src/generate_few_shot_data.py \
--data_type $DATA_TYPE \
--use_semantic_similarity_only \
--num_icl_examples $NUM_EXAMPLES_PER_LABEL \
--model $MODEL \
--outfile $OUTFILE