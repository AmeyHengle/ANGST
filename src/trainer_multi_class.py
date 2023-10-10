import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          EarlyStoppingCallback, EvalPrediction, Trainer,
                          TrainingArguments)

device = "cuda" if torch.cuda.is_available() else "cpu"


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


def training_pipeline(
    model_name,
    text_col,
    max_length,
    batch_size,
    num_epochs,
    metric_name,
    random_state,
    learning_rate,
):
    # Data loading

    df_train = pd.read_csv("../data/silver_data/silver_labels_gpt_3.5_turbo_train.csv")
    df_val = pd.read_csv(
        "../data/silver_data/silver_labels_gpt_3.5_turbo_validation.csv"
    )
    logger.debug(f"Train: {df_train.shape}\nValidation: {df_val.shape}\n")

    df_train["silver_label"].value_counts()
    df_train["depression"] = df_train["silver_label"].apply(
        lambda x: True
        if x in ["Depression", "Comorbid (Depression + Anxiety)"]
        else False
    )
    df_train["anxiety"] = df_train["silver_label"].apply(
        lambda x: True if x in ["Anxiety", "Comorbid (Depression + Anxiety)"] else False
    )

    df_val["silver_label"].value_counts()
    df_val["depression"] = df_val["silver_label"].apply(
        lambda x: True
        if x in ["Depression", "Comorbid (Depression + Anxiety)"]
        else False
    )
    df_val["anxiety"] = df_val["silver_label"].apply(
        lambda x: True if x in ["Anxiety", "Comorbid (Depression + Anxiety)"] else False
    )

    df_test = pd.read_csv("../data/test/full_test.csv")
    df_test["depression"] = df_test["disorder"].apply(
        lambda x: True if "depressive_disorder" in x else False
    )
    df_test["anxiety"] = df_test["disorder"].apply(
        lambda x: True if "anxiety" in x else False
    )

    df_train_dict = {
        "id": df_train["id"].values.tolist(),
        "text": df_train["text"].values.tolist(),
        "depression": df_train["depression"].values.tolist(),
        "anxiety": df_train["anxiety"].values.tolist(),
    }

    df_val_dict = {
        "id": df_val["id"].values.tolist(),
        "text": df_val["text"].values.tolist(),
        "depression": df_val["depression"].values.tolist(),
        "anxiety": df_val["anxiety"].values.tolist(),
    }

    df_test_dict = {
        "id": df_test["id"].values.tolist(),
        "text": df_test["text"].values.tolist(),
        "depression": df_test["depression"].values.tolist(),
        "anxiety": df_test["anxiety"].values.tolist(),
    }

    train_dataset = Dataset.from_dict(df_train_dict)
    val_dataset = Dataset.from_dict(df_val_dict)
    test_dataset = Dataset.from_dict(df_test_dict)

    labels = [
        label for label in test_dataset.features.keys() if label not in ["id", "text"]
    ]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    logger.debug(f"Labels: {labels}")

    # ----------------------------------------------------------------------------------------- #

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_data(examples):
        # take a batch of texts
        text = examples[text_col]
        # encode them
        encoding = tokenizer(
            text, padding="max_length", truncation=True, max_length=max_length
        )
        # add labels
        labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(labels)))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()

        return encoding

    train_dataset_encoded = train_dataset.map(
        preprocess_data, batched=True, remove_columns=train_dataset.column_names
    )
    val_dataset_encoded = val_dataset.map(
        preprocess_data, batched=True, remove_columns=val_dataset.column_names
    )
    test_dataset_encoded = test_dataset.map(
        preprocess_data, batched=False, remove_columns=test_dataset.column_names
    )

    train_dataset_encoded.set_format("torch")
    val_dataset_encoded.set_format("torch")
    test_dataset_encoded.set_format("torch")

    # ----------------------------------------------------------------------------------------- #

    def multi_label_metrics(predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
        roc_auc = roc_auc_score(y_true, y_pred, average="weighted")
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {"f1": f1_weighted, "roc_auc": roc_auc, "accuracy": accuracy}

        return metrics

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        result = multi_label_metrics(predictions=preds, labels=p.label_ids)
        return result

    # ----------------------------------------------------------------------------------------- #

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type="multi_label_classification",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    model.to(device)

    # ----------------------------------------------------------------------------------------- #
    output_dir = f"{model_name}_{max_length}_{num_epochs}_{batch_size}_{learning_rate}"

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        greater_is_better=True,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        no_cuda=False,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset_encoded,
        eval_dataset=train_dataset_encoded,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.01
            )
        ],
    )

    trainer.train()

    # ----------------------------------------------------------------------------------------- #

    # Save model
    best_model_dir = f"{output_dir}/best_model"

    trainer.model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # ----------------------------------------------------------------------------------------- #

    # Run Inference
    texts = df_test["text"].values.tolist()
    predictions_raw, predictions_proc = inference_pipeline(
        texts, model, tokenizer, max_length, id2label
    )
    df_test["labels"] = [np.array(x) for x in test_dataset_encoded["labels"]]
    df_test["predicted_labels"] = predictions_raw
    df_test["predicted_disorder"] = [set(x) for x in predictions_proc]

    df_test[
        [
            "id",
            "text",
            "labels",
            "predicted_labels",
            "disorder",
            "predicted_disorder",
        ]
    ].to_csv(os.path.join(best_model_dir, "predictions.csv"), index=False)


def inference_pipeline(texts, model, tokenizer, max_length, id2label):
    predictions_raw = []
    predictions_proc = []

    for text in tqdm(texts, desc=f"Running inference on {len(texts)} texts"):
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoding = {k: v.to(model.device) for k, v in encoding.items()}

        outputs = model(**encoding)
        logits = outputs.logits

        # apply sigmoid + threshold
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        predictions_raw.append(predictions)
        # turn predicted id's into actual label names
        predicted_labels = [
            id2label[idx] for idx, label in enumerate(predictions) if label == 1.0
        ]
        predictions_proc.append(predicted_labels)

    return predictions_raw, predictions_proc


def main():
    parser = argparse.ArgumentParser(
        description="Command-line arguments for the script"
    )

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--text_col", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--metric_name", type=str, default="f1")
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    # Access the variables using args
    print("Model Name:", args.model_name)
    print("Text Column:", args.text_col)
    print("Max Length:", args.max_length)
    print("Batch Size:", args.batch_size)
    print("Learning Rate:", args.batch_size)
    print("Number of Epochs:", args.num_epochs)
    print("Metric Name:", args.metric_name)
    print("Random State:", args.random_state)

    # Pass the captured arguments to the training_pipeline function
    training_pipeline(
        args.model_name,
        args.text_col,
        args.max_length,
        args.batch_size,
        args.num_epochs,
        args.metric_name,
        args.random_state,
        args.learning_rate,
    )


if __name__ == "__main__":
    main()


"""
python trainer.py \
    --model_name AIMH/mental-bert-base-cased \
    --text_col text \
    --max_length 1 \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --num_epochs 1 \
    --metric_name f1 \
    --random_state 42 \
;

"""
