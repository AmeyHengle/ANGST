import argparse
import gc
import os

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


def training_pipeline(
    model_name,
    label_col,
    text_col,
    max_length,
    batch_size,
    num_epochs,
    metric_name,
    random_state,
    learning_rate,
):
    train_dataset, val_dataset = load_and_preprocess_dataset(
        random_state, model_name, text_col, label_col, max_length
    )

    # 4. Load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=ID2LABEL
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output_dir = f"{model_name.replace('AIMH',label_col)}_{max_length}_{num_epochs}_{batch_size}_{learning_rate}"

    # 5. Fine-tune pretrained model using Huggingface Trainer class
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        greater_is_better=False,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        no_cuda=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3
            )
        ],
    )

    trainer.train()

    # 6. Save model output to specified directory
    best_model_dir = f"{output_dir}/best_model"
    trainer.model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

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
    ].to_csv(os.path.join(best_model_dir, "predictions.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text Classification Training Pipeline"
    )

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--label_col", type=str, required=True)
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--metric_name", type=str, default="eval_loss")
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    # Access the variables using args
    print("Model Name:", args.model_name)
    print("Label Column:", args.label_col)
    print("Text Column:", args.text_col)
    print("Max Length:", args.max_length)
    print("Batch Size:", args.batch_size)
    print("Learning Rate:", args.learning_rate)
    print("Number of Epochs:", args.num_epochs)
    print("Metric Name:", args.metric_name)
    print("Random State:", args.random_state)

    free_memory()
    training_pipeline(
        args.model_name,
        args.label_col,
        args.text_col,
        args.max_length,
        args.batch_size,
        args.num_epochs,
        args.metric_name,
        args.random_state,
        args.learning_rate,
    )
    free_memory()