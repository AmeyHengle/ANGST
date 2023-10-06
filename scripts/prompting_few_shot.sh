#!/bin/bash

SEED=0
DATA_PATH="./data/test/full_test.csv"
RESULT_DIR="./results/few_shot"
NUM_EXAMPLES_PER_LABEL=2

# MODEL="gpt-3.5-turbo"
MODEL="gpt-4"

# choices=[
#     'depression_mards', 
#     'depression_phq9', 
#     'depression_naive', 
#     'depression_mental_llm',
#      'anxiety_bai', 
#      'anxiety_hamilton', 
#      'anxiety_naive', 
#      'anxiety_mental_llm', 
#      'depression_anxiety_comorbidity'
# ]
PROMPT_TYPE="anxiety_hamilton"

CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/prompting_few_shot.py \
--seed $SEED \
--data_path $DATA_PATH \
--model $MODEL \
--prompt_type $PROMPT_TYPE \
--num_examples_per_label $NUM_EXAMPLES_PER_LABEL \
--result_dir $RESULT_DIR > ./logs/zero_shot/zero_shot_${PROMPT_TYPE}_${MODEL}_num_examples_${NUM_EXAMPLES_PER_LABEL}_${SEED}.log &