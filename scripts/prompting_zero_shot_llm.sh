#!/bin/bash

#SBATCH --job-name=inference_exp
#SBATCH --partition=babel-shared-long
#SBATCH --cpus-per-task=2
#SBATCH --gres gpu:4
#SBATCH --mem-per-gpu=40GB
#SBATCH --time=2-23:00:00
#SBATCH --output=logs/zero_shot_llm/zero_shot_comorbidity_mental_flan_t5_v3.log
#SBATCH --error=errors/zero_shot_llm/zero_shot_comorbidity_mental_flan_t5_v3.err

source activate llm_env

DATA_PATH="./data/test/full_test.csv"
RESULT_DIR="./results/zero_shot"
VERSION=3

MODEL="mental_flan_t5"
# MODEL="mental_alpaca"

# PROMPT_TYPE="depression"
# PROMPT_TYPE="anxiety"
PROMPT_TYPE="comorbidity"

python3 src/prompting_zero_shot.py \
--seed $VERSION \
--data_path $DATA_PATH \
--model $MODEL \
--prompt_type $PROMPT_TYPE \
--version $VERSION \
--result_dir $RESULT_DIR