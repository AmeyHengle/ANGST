# /bin/bash
# This script should be excuted in the root directory of the project
# chmod +x scripts/model/train_and_predict.sh
# ./scripts/model/train_and_predict.sh model_name

bash scripts/train_model.sh train_"train_xxl_mh_depression"
bash scripts/predict_model.sh "test_xxl_mh_depression"

bash scripts/train_model.sh train_"train_xxl_mh_anxiety"
bash scripts/predict_model.sh "test_xxl_mh_anxiety"

bash scripts/train_model.sh train_"train_xxl_mh_comorbidity"
bash scripts/predict_model.sh "test_xxl_mh_comorbidity"