#!/bin/bash

DATASET=SDCNL
DATA_PATH=./data/dataset_analysis/${DATASET}.csv
RESULT_DIR=./data/dataset_analysis/v_info/

python3 src/prepare_data_vinfo.py \
--dataset $DATASET \
--fname $DATA_PATH \
--output_dir $RESULT_DIR