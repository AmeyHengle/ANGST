
"""
python test.py \
    --test_file=data/test/test.csv \
    --model_path=runs/2022-11-05_12:27:44/best_model \
    --output_dir=runs/2022-11-05_12:27:44 \
    ;
"""

import logging
import os
import json
import numpy as np
import pandas as pd
from loguru import logger
import argparse
import torch
from utils import clean_text, label_to_word, word_to_label, load_json, save_json
import constants as const
from simpletransformers.classification import (
    MultiLabelClassificationModel, MultiLabelClassificationArgs
)

from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    hamming_loss, 
    f1_score, 
    roc_auc_score, 
    zero_one_loss, 
    coverage_error, 
    label_ranking_loss, 
    average_precision_score
)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
CUDA = torch.cuda.is_available()


def preprocess_dataset(df:pd.DataFrame, text_col:str, label_col:str, normalize_text_col:bool=True)-> pd.DataFrame:
    df[label_col] = df[label_col].apply(lambda x: json.loads(x))
    if normalize_text_col:
        df[text_col] = df[text_col].apply(lambda x: clean_text(x))
        
    df = df.rename(columns={text_col: "text", label_col: "labels"})
    df = df[df["text"].notna()]
    df = df[df["labels"].notna()]
    df = df.reset_index()
    
    return df


def get_metrics(y_true, y_pred):
    
    # --------------- Multi-label clf metrics ---------------
    clf_report_mlc = classification_report(y_true, y_pred, output_dict=True)
    mlc_metrics = {
        "accuracy_score":accuracy_score(y_true, y_pred),
        "hamming_loss":hamming_loss(y_true, y_pred),
        "micro_f1":f1_score(y_true, y_pred, average="micro"),
        "macro_f1":f1_score(y_true, y_pred, average="macro"),
        "weighted_f1":f1_score(y_true, y_pred, average="weighted"),
        "micro_auc":roc_auc_score(y_true, y_pred, average="micro"),
        "macro_auc":roc_auc_score(y_true, y_pred, average="macro"),
        "weighted_auc":roc_auc_score(y_true, y_pred, average="weighted"),
        "one_error":zero_one_loss(y_true, y_pred),
        "coverage":coverage_error(y_true, y_pred),
        "ranking_loss":label_ranking_loss(y_true, y_pred),
        "micro_avg_precision":average_precision_score(y_true, y_pred, average="micro"),
        "macro_avg_precision":average_precision_score(y_true, y_pred, average="macro"),
        "weighted_avg_precision":average_precision_score(y_true, y_pred, average="weighted")
    }


    # --------------- Multi-class clf metrics ---------------
    y_true = [label_to_word(x) for x in y_true]
    y_pred = [label_to_word(x) for x in y_pred]
    clf_report_mcc = classification_report(y_true, y_pred, output_dict=True)

    return mlc_metrics, clf_report_mlc, clf_report_mcc

    
def eval_pipeline(
    model_path: str,
    model_type: str,
    datapath_val: str,
    text_col: str,
    label_col: str,
    output_dir: str
):
    if not os.path.exists(os.path.join(output_dir,"metrics")):
        os.mkdir(os.path.join(output_dir,"metrics"))
    
    # Load train and eval sets
    df_eval = pd.read_csv(datapath_val)
    
    # Load train and eval sets
    df_eval = preprocess_dataset(df_eval,text_col,label_col)
    logger.debug(f"\nEval Set: {df_eval.shape}\n{df_eval[label_col].value_counts()}")

    # Load Model
    model = MultiLabelClassificationModel(
        model_type,
        model_path
    )
    
    # Evaluate on df_eval
    loss, y_pred, wrong_predictions = model.eval_model(df_eval)
    
    # Calculate Metrics
    y_pred = np.rint(y_pred).tolist()
    y_true = df_eval[label_col].values.tolist()
    mlc_metrics, clf_report_mlc, clf_report_mcc = get_metrics(y_true, y_pred)
    
    # Save Metrics
    save_json(loss, os.path.join(output_dir,"metrics","mlc_loss.json"))
    pd.DataFrame().from_dict([mlc_metrics]).T.to_csv(os.path.join(output_dir,"metrics","mlc_metrics.csv"),index=True)
    pd.DataFrame().from_dict(clf_report_mlc).T.to_csv(os.path.join(output_dir,"metrics","clf_report_mlc.csv"),index=True)
    pd.DataFrame().from_dict(clf_report_mcc).T.to_csv(os.path.join(output_dir,"metrics","clf_report_mcc.csv"),index=True)  

    # Save predictions
    predictions_df = df_eval[[text_col]]
    predictions_df['y_true'] = y_true
    predictions_df['y_pred'] = y_pred
    predictions_df.to_csv(os.path.join(output_dir,"metrics","predictions.csv"),index=False)    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file")
    parser.add_argument("--model_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--text_col", default=const.TEXT_COL)
    parser.add_argument("--label_col", default=const.LABEL_COL)
    parser.add_argument("--model_type", default=const.MODEL_TYPE)

    args = parser.parse_args()

    datapath_val = args.test_file
    model_path = args.model_path
    text_col = args.text_col
    label_col = args.label_col
    model_type = args.model_type
    output_dir = args.output_dir
    
    eval_pipeline(
        model_path,
        model_type,
        datapath_val,
        text_col,
        label_col,
        output_dir
    )