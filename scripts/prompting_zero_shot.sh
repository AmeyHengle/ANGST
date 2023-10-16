#!/bin/bash

DATA_PATH="./data/test/full_test.csv"
RESULT_DIR="./results/zero_shot"
VERSION=3

MODEL="gpt-3.5-turbo"
# MODEL="gpt-4"
# MODEL="mental_flan_t5"
# MODEL="mental_alpaca"


# choices=[
#     'depression_mards', 
#     'depression_phq9', 
#     'depression', 
#     'depression_mental_llm',
#      'anxiety_bai', 
#      'anxiety_hamilton', 
#      'anxiety', 
#      'anxiety_mental_llm', 
#      'comorbidity'
# ]
PROMPT_TYPE="depression_mards"

# python3 src/prompting_zero_shot.py \
# --seed $VERSION \
# --data_path $DATA_PATH \
# --model $MODEL \
# --prompt_type $PROMPT_TYPE \
# --version $VERSION \
# --result_dir $RESULT_DIR

nohup python3 -u src/prompting_zero_shot.py \
--seed $VERSION \
--data_path $DATA_PATH \
--model $MODEL \
--prompt_type $PROMPT_TYPE \
--version $VERSION \
--result_dir $RESULT_DIR > ./logs/zero_shot/zero_shot_${PROMPT_TYPE}_${MODEL}_version_${VERSION}_seed_${VERSION}.log &