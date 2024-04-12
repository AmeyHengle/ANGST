#!/bin/bash

#SBATCH --job-name=llm_inference_exp
#SBATCH --partition=general
#SBATCH --time=1-23:58:00
#SBATCH --gres gpu:A6000:1
#SBATCH --mem-per-gpu=30GB
#SBATCH --output=logs/zero_shot_llm/anxiety_llama_chat_7b_v1.log
#SBATCH --error=errors/zero_shot_llm/anxiety_llama_chat_7b_v1.err

DATA_PATH="./data/test/full_test.csv"
RESULT_DIR="./results/zero_shot"
VERSION=1

# MODEL="mental_llama_chat_7b"
# MODEL="mental_llama_chat_13b"
MODEL="llama_chat_7b" 
# MODEL="llama_chat_13b"
# MODEL="llama_chat_70b"


# PROMPT_TYPE="depression_llama"
PROMPT_TYPE="anxiety_llama"

python3 src/prompting_zero_shot.py \
--seed $VERSION \
--data_path $DATA_PATH \
--model $MODEL \
--prompt_type $PROMPT_TYPE \
--version $VERSION \
--result_dir $RESULT_DIR