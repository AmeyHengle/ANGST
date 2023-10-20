# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/train_model.sh
# ./scripts/train_model.sh


EXP="$1"

if [ -z " $EXP " ] ; then
    echo "Please specify the experiment name"
    exit 1
fi

if [[ $EXP == "train_xxl" ]]; then
    python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_xxl_2_epoch.gin" --gin.MODEL_DIR="'checkpoints/flant5-model-xxl'" --gin.MODE="'train'"
elif [[ $EXP == "train_xl_w_o_context" ]]; then
    python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_xl_2_epoch.gin" --gin_file="exp/configs/without_context.gin" --gin.MODEL_DIR="'checkpoints/flant5-model-xl-w-o-cotext'" --gin.MODE="'train'"
elif [[ $EXP == "train_xl" ]]; then
    python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_xl_2_epoch.gin" --gin.MODEL_DIR="'checkpoints/flant5-model-xl'" --gin.MODE="'train'"
elif [[ $EXP == "train_large" ]]; then
    python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_large_2_epoch.gin" --gin.MODEL_DIR="'checkpoints/flant5-model-large'" --gin.MODE="'train'"
elif [[ $EXP == "train_base" ]]; then
    python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_base_2_epoch.gin" --gin.MODEL_DIR="'checkpoints/flant5-model-base'" --gin.MODE="'train'"
elif [[ $EXP == "train_small" ]]; then
    python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_small_2_epoch.gin" --gin.MODEL_DIR="'checkpoints/flant5-model-small'" --gin.MODE="'train'" --gin.SPLIT="'comorbidity'"

#---------------------------------------------------------------------------------------------#
# Mental-Health experiments
#---------------------------------------------------------------------------------------------#

elif [[ $EXP == "train_xxl_mh_depression" ]]; then
    python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_mental_flant5_xxl_3_epoch.gin" --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-depression'" --gin.MODE="'train'" --gin.SPLIT="'depression'"
elif [[ $EXP == "train_xxl_mh_anxiety" ]]; then
    python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_mental_flant5_xxl_3_epoch.gin" --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-anxiety'" --gin.MODE="'train'" --gin.SPLIT="'anxiety'"
elif [[ $EXP == "train_xxl_mh_comorbidity" ]]; then
    python src/train.py --gin_file="scripts/model.gin" --gin_file="exp/configs/train_model_mental_flant5_xxl_3_epoch.gin" --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-comorbidity'" --gin.MODE="'train'" --gin.SPLIT="'comorbidity'"
