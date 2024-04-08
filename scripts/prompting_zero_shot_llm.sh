#!/bin/bash

#SBATCH --job-name=llm_inference_exp
#SBATCH --partition=general
#SBATCH --time=1-23:58:00
#SBATCH --gres gpu:A6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=40GB
#SBATCH --output=logs/zero_shot_llm/depression_mental_llama_chat_7b.log
#SBATCH --error=errors/zero_shot_llm/depression_mental_llama_chat_7b.err

DATA_PATH="./data/test/full_test.csv"
RESULT_DIR="./results/zero_shot"
VERSION=1

MODEL="mental_llama_chat_7b"
# MODEL="mental_llama_chat_13b"

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
PROMPT_TYPE="depression"

python3 src/prompting_zero_shot.py \
--seed $VERSION \
--data_path $DATA_PATH \
--model $MODEL \
--prompt_type $PROMPT_TYPE \
--version $VERSION \
--result_dir $RESULT_DIR