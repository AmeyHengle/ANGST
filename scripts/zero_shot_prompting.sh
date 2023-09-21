#!/bin/bash

SEED=0
DATA_PATH="./data/test/full_test.csv"
RESULT_DIR="./results/zero_shot"
MODEL="gpt-3.5-turbo"
# MODEL="gpt-4"

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
PROMPT_TYPE="depression_mards"

CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/zero_shot_prompting.py \
--seed $SEED \
--data_path $DATA_PATH \
--model $MODEL \
--prompt_type $PROMPT_TYPE \
--result_dir $RESULT_DIR > ./logs/zero_shot/zero_shot_${PROMPT_TYPE}_${MODEL}_${SEED}.log &