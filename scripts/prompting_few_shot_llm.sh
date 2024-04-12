#!/bin/bash

#SBATCH --job-name=few_shot_llm_inference_exp
#SBATCH --partition=general
#SBATCH --time=1-23:58:00
#SBATCH --gres gpu:A6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-gpu=40GB
#SBATCH --output=logs/few_shot_llm/depression_llama_chat_7b_v2.log
#SBATCH --error=errors/few_shot_llm/depression_llama_chat_7b_v2.err


RESULT_DIR="./results/few_shot"
VERSION=2

PROMPT_TYPE="depression"
# PROMPT_TYPE="anxiety"
# PROMPT_TYPE="comorbidity"

# Openai prompting
# NUM_EXAMPLES_PER_LABEL=4
# MODEL="gpt-3.5-turbo"
# # MODEL="gpt-4"
# DATA_PATH=./data/few_shot_prompts/${PROMPT_TYPE}_num_examples_ss_${NUM_EXAMPLES_PER_LABEL}.csv

# LLama Prompting
NUM_EXAMPLES_PER_LABEL=2
MODEL="llama_chat_7b"
# MODEL="mental_llama_chat_7b"
# MODEL="mental_llama_chat_13b"
DATA_PATH=./data/few_shot_prompts/${PROMPT_TYPE}_llama_num_examples_ss_${NUM_EXAMPLES_PER_LABEL}.csv

python3 src/prompting_few_shot.py \
--seed $VERSION \
--data_path $DATA_PATH \
--model $MODEL \
--prompt_type $PROMPT_TYPE \
--version $VERSION \
--result_dir $RESULT_DIR