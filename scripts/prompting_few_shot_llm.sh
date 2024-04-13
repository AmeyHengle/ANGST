#!/bin/bash

#SBATCH --job-name=fs_llm_inference_exp
#SBATCH --partition=general
#SBATCH --time=1-23:58:00
#SBATCH --gres gpu:A6000:2
#SBATCH --mem-per-gpu=40GB
#SBATCH --output=logs/few_shot_llm/anxiety_llama_chat_13b_v3_gpt_prompt.log
#SBATCH --error=errors/few_shot_llm/anxiety_llama_chat_13b_v3_gpt_prompt.err

# MODEL="llama_chat_7b" 
MODEL="llama_chat_13b"
# MODEL="llama_chat_70b"

# PROMPT_TYPE="depression"
PROMPT_TYPE="anxiety"
# PROMPT_TYPE="comorbidity"

NUM_EXAMPLES_PER_LABEL=2
DATA_PATH=./data/few_shot_prompts/${PROMPT_TYPE}_num_examples_ss_${NUM_EXAMPLES_PER_LABEL}.csv
RESULT_DIR="./results/few_shot"
VERSION=3

python3 src/prompting_few_shot.py \
--seed $VERSION \
--data_path $DATA_PATH \
--model $MODEL \
--prompt_type $PROMPT_TYPE \
--num_examples_per_label $NUM_EXAMPLES_PER_LABEL \
--version $VERSION \
--result_dir $RESULT_DIR