#!/bin/bash

#SBATCH --job-name=inference_exp
#SBATCH --partition=babel-shared-long
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:4
#SBATCH --mem-per-gpu=40GB
#SBATCH --time=2-23:00:00
#SBATCH --output=logs/zero_shot_llm/zero_shot_anxiety_hamilton_mental_flan_t5_v1.log
#SBATCH --error=errors/zero_shot_llm/zero_shot_anxiety_hamilton_mental_flan_t5_v1.err

source activate llm_env

DATA_PATH="./data/test/full_test.csv"
RESULT_DIR="./results/zero_shot"
VERSION=1

MODEL="mental_flan_t5"
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
# PROMPT_TYPE="depression"
# PROMPT_TYPE="anxiety"
PROMPT_TYPE="anxiety_hamilton"

python3 src/prompting_zero_shot.py \
--seed $VERSION \
--data_path $DATA_PATH \
--model $MODEL \
--prompt_type $PROMPT_TYPE \
--version $VERSION \
--result_dir $RESULT_DIR