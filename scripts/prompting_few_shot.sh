#!/bin/bash

SEED=0
RESULT_DIR="./results/few_shot"
NUM_EXAMPLES_PER_LABEL=1

# MODEL="gpt-3.5-turbo"
MODEL="gpt-4"

# PROMPT_TYPE choices=[
#     'depression', 
#      'anxiety', 
#      'comorbidity'
# ]
PROMPT_TYPE="comorbidity"

DATA_PATH=./data/few_shot_prompts/${PROMPT_TYPE}_num_examples_per_label_${NUM_EXAMPLES_PER_LABEL}.csv


CUDA_VISIBLE_DEVICES=2 nohup python3 -u src/prompting_few_shot.py \
--seed $SEED \
--data_path $DATA_PATH \
--model $MODEL \
--prompt_type $PROMPT_TYPE \
--num_examples_per_label $NUM_EXAMPLES_PER_LABEL \
--result_dir $RESULT_DIR > ./logs/few_shot/few_shot_${PROMPT_TYPE}_${MODEL}_num_examples_${NUM_EXAMPLES_PER_LABEL}_${SEED}.log &