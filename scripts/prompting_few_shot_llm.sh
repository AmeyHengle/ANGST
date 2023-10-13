#!/bin/bash

#SBATCH --job-name=inference_exp
#SBATCH --partition=babel-shared-long
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:4
#SBATCH --mem-per-gpu=40GB
#SBATCH --time=2-23:00:00
#SBATCH --output=logs/few_shot_llm/few_shot_depression_mental_alpaca_t5_v1.log
#SBATCH --error=errors/few_shot_llm/few_shot_depression_mental_alpaca_t5_v1.err

source activate llm_env

RESULT_DIR="./results/few_shot"
NUM_EXAMPLES_PER_LABEL=4
VERSION=1

PROMPT_TYPE="depression"
# PROMPT_TYPE="anxiety"
# PROMPT_TYPE="comorbidity"

# MODEL="mental_flan_t5"
MODEL="mental_alpaca"

DATA_PATH=./data/few_shot_prompts/${PROMPT_TYPE}_num_examples_ss_${NUM_EXAMPLES_PER_LABEL}.csv

python3 src/prompting_few_shot.py \
--seed $VERSION \
--data_path $DATA_PATH \
--model $MODEL \
--prompt_type $PROMPT_TYPE \
--version $VERSION \
--result_dir $RESULT_DIR