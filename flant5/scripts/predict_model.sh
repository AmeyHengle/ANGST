# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/predict_model.sh
# ./scripts/predict_model.sh [experiment_name]
## Default evaluate metric is BLEU only

EXP="$1"
EVALUATE_METRICS="['bleu', 'bertscore', 'rouge']"

if [ -z " $EXP " ] ; then
    echo "Please specify the experiment name"
    exit 1
fi

if [[ $EXP == "xxl" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-xxl'" \
        --gin.EVALUATE_METRICS="['bleu', 'bertscore', 'rouge']" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-xxl/results.csv'" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl_greedy" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin_file="exp/configs/greedy.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-xl'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-xl/greedy-results.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl_topp" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin_file="exp/configs/topp.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-xl'" \
        --gin.OUTPUT_DIR="'checkpoints/flant5-model-xl/topp'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-xl/topp-results.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl_topk" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin_file="exp/configs/topk.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-xl'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-xl/topk-results.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl_w_o_context" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin_file="exp/configs/without_context.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-xl-w-o-cotext'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-xl-w-o-cotext/results.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "xl" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-xl'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-xl/results.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "large" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-large'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-large/results.csv'" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=4
elif [[ $EXP == "base" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-base'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-base/results.csv'" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=16
elif [[ $EXP == "small" ]]; then
    CUDA_LAUNCH_BLOCKING=1 python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-small'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-small/results.csv'" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=16 \
        --gin.SPLIT="'comorbidity'"

elif [[ $EXP == "xl_adv" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin_file="exp/configs/adv_context.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-xl'" \
        --gin.OUTPUT_DIR="'checkpoints/flant5-model-xl/adv'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-xl/results_adv.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=4

    python tools/src/evaluate_advContext.py \
        --prediction_file "checkpoints/flant5-model-xl/adv/answer.csv"
elif [[ $EXP == "xl_topp_adv" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin_file="exp/configs/topp.gin" \
        --gin_file="exp/configs/adv_context.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-xl'" \
        --gin.OUTPUT_DIR="'checkpoints/flant5-model-xl/adv-topp'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-xl/results_adv_topp.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=16

    python tools/src/evaluate_advContext.py \
        --prediction_file "checkpoints/flant5-model-xl/adv-topp/answer.csv"
elif [[ $EXP == "xl_wo_context_adv" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin_file="exp/configs/adv_context.gin" \
        --gin_file="exp/configs/without_context.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-xl-w-o-cotext'" \
        --gin.OUTPUT_DIR="'checkpoints/flant5-model-xl-w-o-cotext/adv'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-xl-w-o-cotext/results_adv.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=4

    python tools/src/evaluate_advContext.py \
        --prediction_file "checkpoints/flant5-model-xl-w-o-cotext/adv/answer.csv"
elif [[ $EXP == "xl_wo_context_topp_adv" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin_file="exp/configs/topp.gin" \
        --gin_file="exp/configs/adv_context.gin" \
        --gin_file="exp/configs/without_context.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-xl-w-o-cotext'" \
        --gin.OUTPUT_DIR="'checkpoints/flant5-model-xl-w-o-cotext/adv-topp'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-xl-w-o-cotext/results_adv_topp.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=16

    python tools/src/evaluate_advContext.py \
        --prediction_file "checkpoints/flant5-model-xl-w-o-cotext/adv-topp/answer.csv"
elif [[ $EXP == "xxl_adv_fp16" ]]; then
    # python src/inference.py \
    #     --gin_file="scripts/model_inference.gin" \
    #     --gin_file="exp/configs/adv_context.gin" \
    #     --gin_file="scripts/fp16.gin" \
    #     --gin.MODEL_DIR="'checkpoints/flant5-model-xxl'" \
    #     --gin.OUTPUT_DIR="'checkpoints/flant5-model-xxl/adv'" \
    #     --gin.RESULT_FILE="'checkpoints/flant5-model-xxl/results_adv.csv'" \
    #     --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
    #     --gin.MODE="'test'" \
    #     --gin.BATCH_SIZE=4

    python tools/src/evaluate_advContext.py \
        --prediction_file "checkpoints/flant5-model-xxl/adv/answer.csv"
elif [[ $EXP == "small_adv" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin_file="exp/configs/adv_context.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-small'" \
        --gin.OUTPUT_DIR="'checkpoints/flant5-model-small/adv'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-small/results_adv.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=16

    python tools/src/evaluate_advContext.py \
        --prediction_file "checkpoints/flant5-model-small/adv/answer.csv"
elif [[ $EXP == "large_adv" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin_file="exp/configs/adv_context.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-large'" \
        --gin.OUTPUT_DIR="'checkpoints/flant5-model-large/adv'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-large/results_adv.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=16

    python tools/src/evaluate_advContext.py \
        --prediction_file "checkpoints/flant5-model-large/adv/answer.csv"
elif [[ $EXP == "base_adv" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin_file="exp/configs/adv_context.gin" \
        --gin.MODEL_DIR="'checkpoints/flant5-model-base'" \
        --gin.OUTPUT_DIR="'checkpoints/flant5-model-base/adv'" \
        --gin.RESULT_FILE="'checkpoints/flant5-model-base/results_adv.csv'" \
        --gin.EVALUATE_METRICS="$EVALUATE_METRICS" \
        --gin.MODE="'test'" \
        --gin.BATCH_SIZE=16

    python tools/src/evaluate_advContext.py \
        --prediction_file "checkpoints/flant5-model-base/adv/answer.csv"

#---------------------------------------------------------------------------------------------#
# Mental-Health experiments
#---------------------------------------------------------------------------------------------#

elif [[ $EXP == "test_xxl_mh_depression" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-depression'" \
        --gin.EVALUATE_METRICS="['bleu', 'bertscore', 'rouge']" \
        --gin.RESULT_FILE="'checkpoints/mental-flant5-xxl-depression/results.csv'" \
        --gin.MODE="'test'" \
        --gin.SPLIT="'depression'" \
        --gin.BATCH_SIZE=16

elif [[ $EXP == "test_xxl_mh_anxiety" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-anxiety'" \
        --gin.EVALUATE_METRICS="['bleu', 'bertscore', 'rouge']" \
        --gin.RESULT_FILE="'checkpoints/mental-flant5-xxl-anxiety/results.csv'" \
        --gin.MODE="'test'" \
        --gin.SPLIT="'anxiety'" \
        --gin.BATCH_SIZE=16

elif [[ $EXP == "test_xxl_mh_comorbidity" ]]; then
    python src/inference.py \
        --gin_file="scripts/model_inference.gin" \
        --gin.MODEL_DIR="'checkpoints/mental-flant5-xxl-comorbidity'" \
        --gin.EVALUATE_METRICS="['bleu', 'bertscore', 'rouge']" \
        --gin.RESULT_FILE="'checkpoints/mental-flant5-xxl-comorbidity/results.csv'" \
        --gin.MODE="'test'" \
        --gin.SPLIT="'comorbidity'" \
        --gin.BATCH_SIZE=16

else
    echo "Experiment name not found"
    exit 1
fi
