# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/model/train_and_predict.sh
# ./scripts/model/train_and_predict.sh model_name

EXP="$1"

if [ "$EXP" != "small" ] && [ "$EXP" != "base" ] && [ "$EXP" != "large" ] && [ "$EXP" != "xl" ] && [ "$EXP" != "xxl" ] && [ "$EXP" != "xl_w_o_context" ] ; then
    echo "Experiment name not found"
    exit 1
fi

bash scripts/train_model.sh train_$EXP
bash scripts/predict_model.sh $EXP