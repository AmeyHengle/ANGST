# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/model/train_and_predict.sh
# ./scripts/model/train_and_predict.sh model_name

python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_mental_flant5_xxl_3_epoch.gin" --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-depression'" --gin.MODE="'train'" --gin.SPLIT="'depression'"
python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_mental_flant5_xxl_3_epoch.gin" --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-anxiety'" --gin.MODE="'train'" --gin.SPLIT="'anxiety'"
python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_mental_flant5_xxl_3_epoch.gin" --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-comorbidity'" --gin.MODE="'train'" --gin.SPLIT="'comorbidity'"


python src/inference.py \
    --gin_file="scripts/model_inference.gin" \
    --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-depression'" \
    --gin.EVALUATE_METRICS="['bleu', 'bertscore', 'rouge']" \
    --gin.RESULT_FILE="'checkpoints/mental-flant5-xxl-depression/results.csv'" \
    --gin.MODE="'test'" \
    --gin.SPLIT="'depression'" \
    --gin.BATCH_SIZE=16

python src/inference.py \
    --gin_file="scripts/model_inference.gin" \
    --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-anxiety'" \
    --gin.EVALUATE_METRICS="['bleu', 'bertscore', 'rouge']" \
    --gin.RESULT_FILE="'checkpoints/mental-flant5-xxl-anxiety/results.csv'" \
    --gin.MODE="'test'" \
    --gin.SPLIT="'anxiety'" \
    --gin.BATCH_SIZE=16

python src/inference.py \
    --gin_file="scripts/model_inference.gin" \
    --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-comorbidity'" \
    --gin.EVALUATE_METRICS="['bleu', 'bertscore', 'rouge']" \
    --gin.RESULT_FILE="'checkpoints/mental-flant5-xxl-comorbidity/results.csv'" \
    --gin.MODE="'test'" \
    --gin.SPLIT="'comorbidity'" \
    --gin.BATCH_SIZE=16