#!/bin/bash

#SBATCH --job-name=hp_exp
#SBATCH --partition=babel-shared-long
#SBATCH --array=1-64%8
#SBATCH --mem=40GB
#SBATCH --time=2-23:00:00
#SBATCH --gres gpu:A6000:1
#SBATCH --output=logs/trainer_binary/trainer_binary_array_job_%A_%a.log
#SBATCH --error=errors/trainer_binary/trainer_binary_array_job_%A_%a.err

source activate llm_env

params=$(tail -n+${SLURM_ARRAY_TASK_ID} hparams_files/trainer_erm_binary_missing_models.txt | head -n1)   
$params
# params=$(tail -n+${SLURM_ARRAY_TASK_ID} hparams_files/trainer_erm_binary.txt | head -n1)   
# $params