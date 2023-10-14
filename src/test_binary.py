import argparse
import gc
import os
import json

import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, EarlyStoppingCallback)

device = "cuda" if torch.cuda.is_available() else "cpu"
ID2LABEL = {0: 0, 1: 1}


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    id_col: str,
    stratify_col: str,
    random_state: int,
):
    x_train, x_test, y_train, y_test = train_test_split(
        df[id_col].values.tolist(),
        df[stratify_col].values.tolist(),
        stratify=df[stratify_col].values.tolist(),
        test_size=test_size,
        random_state=random_state,
    )

    df_train = df[df[id_col].isin(x_train)]
    df_test = df[df[id_col].isin(x_test)]
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()

    logger.debug(
        f"\nTrain set: {df_train.shape}\n{df_train[stratify_col].value_counts()}"
    )
    logger.debug(f"\nTest Set: {df_test.shape}\n{df_test[stratify_col].value_counts()}")

    return df_train, df_test


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    return {"f1": f1_score(labels, predictions, average="weighted")}


def free_memory():
    torch.cuda.empty_cache()
    gc.collect()


def load_and_preprocess_dataset(
    random_state, model_name, text_col, label_col, max_length
):
    df_train = pd.read_csv("./data/silver_data/silver_labels_gpt_3.5_turbo_train.csv")
    df_val = pd.read_csv(
        "./data/silver_data/silver_labels_gpt_3.5_turbo_validation.csv"
    )
    logger.debug(f"Train: {df_train.shape}\nValidation: {df_val.shape}\n")

    # Tokenizing and converting train and validation datasets separately
    train_dataset = _tokenize_and_convert(
        df_train, model_name, text_col, label_col, max_length
    )
    val_dataset = _tokenize_and_convert(
        df_val, model_name, text_col, label_col, max_length
    )

    return train_dataset, val_dataset


def _tokenize_and_convert(df, model_name, text_col, label_col, max_length):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_inputs = tokenizer(
        df[text_col].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    encoder = OneHotEncoder(sparse=False, categories=[list(ID2LABEL.values())])

    labels_one_hot = encoder.fit_transform(df[[label_col]])

    labels = [list(row).index(1) for row in labels_one_hot]

    dataset = Dataset.from_dict({k: v for k, v in tokenized_inputs.items()})
    dataset = dataset.with_format("torch", columns=["input_ids", "attention_mask"])
    dataset = dataset.add_column("labels", labels)

    id2label = {i: label for i, label in enumerate(encoder.categories_[0])}
    label2id = {label: i for i, label in enumerate(encoder.categories_[0])}

    logger.debug(f"Id2Label: {id2label}")
    logger.debug(f"Label2Id: {label2id}")

    return dataset


def inference_pipeline(texts, model, tokenizer, max_length):
    free_memory()
    y_pred = []
    batch_size = 1

    for i in tqdm(
        range(0, len(texts), batch_size),
        desc=f"Running Binary CLF inference on {len(texts)} data points",
    ):
        batch_inputs = texts[i : i + batch_size]
        tokenized_inputs = tokenizer(
            batch_inputs,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized_inputs.to(device)
        outputs = model(**tokenized_inputs)
        probs = outputs.logits.softmax(dim=1)
        predictions = probs.argmax(dim=1).tolist()
        y_pred.extend(predictions)

    free_memory()

    return y_pred


def run_tests(
    model_dir,
    label_col,
    text_col,
    output_dir
):
    best_model_dir = f"{model_dir}/best_model"
    model_name = model_dir.split('/')[-1].strip()
    
    with open(os.path.join(best_model_dir, 'tokenizer_config.json'), 'r') as f:
        config = json.load(f)
        max_length = int(config['max_length'])

    # 4. Load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        best_model_dir, num_labels=2, id2label=ID2LABEL
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)


    # 7. Run inference on test set and save predictions to a CSV
    df_test = pd.read_csv("./data/test/full_test.csv")
    texts = df_test[text_col].values.tolist()
    predictions = inference_pipeline(texts, model, tokenizer, max_length)
    df_test[f"predicted_{label_col}"] = predictions
    df_test[
        [
            "id",
            "text",
            label_col,
            f"predicted_{label_col}",
        ]
    ].to_csv(os.path.join(output_dir, f"predictions_{model_name}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text Classification Training Pipeline"
    )

    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--label_col", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    # Access the variables using args
    print("Model Dir:", args.model_dir)
    print("Label Column:", args.label_col)
    print("Text Column:", args.text_col)
    print("Output Dir:", args.output_dir)

    free_memory()
    run_tests(
        args.model_dir,
        args.label_col,
        args.text_col,
        args.output_dir,
    )
    free_memory()
    

"""
python src/test_binary.py \
    --model_dir /home/ameyh/mental-health-comorbitidy-classification/binary-clf/anxiety_label/mental-bert-base-cased_12_10_128_2e-05 \
    --label_col "anxiety_label" \
    --text_col "text" \
    --output_dir /home/ameyh/depository/mental-health-comorbitidy-classification
"""