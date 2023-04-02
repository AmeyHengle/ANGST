"""
python train.py \
    --train_file=data/train/v0.1.4.csv \
    --val_file=data/test/test.csv \
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
import datetime
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
    df = df.rename(columns={text_col: "text", label_col: "labels"})
    df = df[df["text"].notna()]
    df = df[df["labels"].notna()]
    
    df[label_col] = df[label_col].apply(lambda x: json.loads(x))
    if normalize_text_col:
        df[text_col] = df[text_col].apply(lambda x: clean_text(x))
        
    if 'level_0' in df.columns:
        df = df.drop(['level_0'],axis=1)
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
    y_true_word = [label_to_word(x) for x in y_true]
    y_pred_word = [label_to_word(x) for x in y_pred]
    clf_report_mcc = classification_report(y_true_word, y_pred_word, output_dict=True)

    return mlc_metrics, clf_report_mlc, clf_report_mcc


def train_pipeline(
    datapath_train: pd.DataFrame,
    datapath_eval: pd.DataFrame,
    output_dir: str,
    classification_args: dict,
    text_col: str,
    label_col: str,
    model_name: str, 
    model_type: str
):
    if output_dir == const.ROOT:
        # run = str(datetime.datetime.now()).split('.')[0].replace(" ","_")
        run = datapath_train.split("/")[-1].replace(".csv","")
        model_dir = os.path.join(output_dir, "models", run)
        results_dir = os.path.join(output_dir, "results", run)        
        os.mkdir(model_dir) 
        os.mkdir(results_dir)
    if not os.path.exists(os.path.join(results_dir,"metrics")):
        os.mkdir(os.path.join(results_dir,"metrics"))
    if not os.path.exists(os.path.join(model_dir,"best_model")):
        os.mkdir(os.path.join(model_dir,"best_model"))

    classification_args['output_dir'] = os.path.join(model_dir,"best_model")
    classification_args['best_model_dir'] = os.path.join(model_dir,"best_model")
    classification_args['manual_seed'] = const.RANDOM_STATE
        
    # Load train and eval sets
    df_train = pd.read_csv(datapath_train)
    df_eval = pd.read_csv(datapath_eval)
    
    df_train = preprocess_dataset(df_train,text_col,label_col)
    df_eval = preprocess_dataset(df_eval,text_col,label_col)
    
    df_train = df_train
    df_eval = df_eval
    
    logger.debug(f"\nTrain set: {df_train.shape}\n{df_train[label_col].value_counts()}")
    logger.debug(f"\nEval Set: {df_eval.shape}\n{df_eval[label_col].value_counts()}")
    
    # Initialize model    
    model_args = MultiLabelClassificationArgs(**classification_args)
    model = MultiLabelClassificationModel(
        model_name=model_name,
        model_type=model_type,
        use_cuda=CUDA,
        args=model_args
    )
    
    # Train
    train_logs = model.train_model(
                    train_df=df_train,
                    eval_df=df_eval
                )
    
    # Evaluate on df_eval
    loss, y_pred, wrong_predictions = model.eval_model(df_eval)
    
    # Calculate Metrics
    y_pred = np.rint(y_pred).tolist()
    y_true = df_eval[label_col].values.tolist()
    mlc_metrics, clf_report_mlc, clf_report_mcc = get_metrics(y_true, y_pred)
    
    # Save Metrics
    save_json(train_logs, os.path.join(results_dir,"metrics","train_logs.json"))
    save_json(loss, os.path.join(results_dir,"metrics","mlc_loss.json"))
    pd.DataFrame().from_dict([mlc_metrics]).T.to_csv(os.path.join(results_dir,"metrics","mlc_metrics.csv"),index=True)
    pd.DataFrame().from_dict(clf_report_mlc).T.to_csv(os.path.join(results_dir,"metrics","clf_report_mlc.csv"),index=True)
    pd.DataFrame().from_dict(clf_report_mcc).T.to_csv(os.path.join(results_dir,"metrics","clf_report_mcc.csv"),index=True)
    
    # Save predictions
    predictions_df = df_eval[[text_col]]
    predictions_df['y_true'] = y_true
    predictions_df['y_pred'] = y_pred
    predictions_df.to_csv(os.path.join(results_dir,"metrics","predictions.csv"),index=False)
    
    
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file")
    parser.add_argument("--val_file")
    parser.add_argument("--output_dir", default=const.ROOT)
    parser.add_argument("--classification_args", default=const.CONFIG_TRAIN)
    parser.add_argument("--text_col", default=const.TEXT_COL)
    parser.add_argument("--label_col", default=const.LABEL_COL)
    parser.add_argument("--model_name", default=const.MODEL_NAME)
    parser.add_argument("--model_type", default=const.MODEL_TYPE)
    
    args = parser.parse_args()

    datapath_train = args.train_file
    datapath_val = args.val_file
    text_col = args.text_col
    label_col = args.label_col
    output_dir = args.output_dir
    model_name = args.model_name
    model_type = args.model_type
    classification_args = args.classification_args
    
    train_pipeline(
        datapath_train,
        datapath_val,
        output_dir,
        classification_args,
        text_col,
        label_col,
        model_name, 
        model_type
    )
