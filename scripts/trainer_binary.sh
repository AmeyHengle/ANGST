#!/bin/bash

#SBATCH --job-name=hp_exp
#SBATCH --nodelist=matrix-2-37
#SBATCH --partition=talwalkar_reserved
#SBATCH --gres gpu:1
#SBATCH --output=logs/trainer_binary/trainer_binary_array_job_%A_%a.log
#SBATCH --error=errors/trainer_binary/trainer_binary_array_job_%A_%a.err

python3 src/trainer_binary.py --model_name AIMH/mental-bert-base-cased --text_col text --label_col depression_label --max_length 512 --batch_size 16 --learning_rate 2e-5 --num_epochs 30 --metric_name f1 --random_state 1 

# params=$(tail -n+${SLURM_ARRAY_TASK_ID} hparams_files/trainer_erm_binary.txt | head -n1)   
# $params