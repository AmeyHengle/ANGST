from __future__ import annotations

import json
import os
import re
from functools import partial
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import tqdm
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    DataCollatorForSeq2Seq,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
)
from transformers.integrations import WandbCallback

from src import BaseSBFModel
from src.metrics import (
    aggregated_metrics,
    aggregated_metrics_with_postprocess,
    bleu_metrics,
    prediction_metrics,
)
from src.prompt_templates import (
    map_dataset_to_tokenized_prompt,
)

os.environ["WANDB_PROJECT"] = "flant5"


def log(self: Trainer, logs: Dict[str, float]) -> None:
    """
    override the log method of the Trainer class to
    remove any non-serializable objects from the logs

    Args:
        logs (`Dict[str, float]`):
            The values to log.
    """
    if self.state.epoch is not None:
        logs["epoch"] = round(self.state.epoch, 2)

    output = {**logs, **{"step": self.state.global_step}}

    # edit starts here
    keys_to_remove = []
    for key, value in output.items():
        try:
            _ = json.dumps(value)
        except TypeError:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del output[key]
    # edit ends here

    self.state.log_history.append(output)
    self.control = self.callback_handler.on_log(
        self.args, self.state, self.control, logs
    )


Seq2SeqTrainer.log = log


class Model(BaseSBFModel):
    def __init__(
        self,
        t5_model_name: str = "google/flan-t5-small",
        from_local: bool = False,
        torch_dtype: torch.dtype = torch.float32,
        model_parallelize_for_larger_models: bool = True,
    ):
        assert (
            "t5" in t5_model_name or from_local
        ), "Explain model only supports T5 models."
        try:
            self.model: T5ForConditionalGeneration = cast(
                T5ForConditionalGeneration,
                T5ForConditionalGeneration.from_pretrained(
                    t5_model_name, torch_dtype=torch_dtype
                ),
            )
        except Exception as e:
            print(f"Error loading model {t5_model_name}: {e}")
            raise e

        self.model: T5ForConditionalGeneration = cast(  # type: ignore
            T5ForConditionalGeneration, self.model
        )
        if model_parallelize_for_larger_models and (
            t5_model_name in ["google/flan-t5-xxl", "google/flan-t5-xl"]
            or (from_local and "xl" in t5_model_name)
        ):
            self.model.parallelize()
        self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
            t5_model_name
        )

    def train(
        self,
        dataset: DatasetDict,
        args: Seq2SeqTrainingArguments = Seq2SeqTrainingArguments(
            output_dir="checkpoints/_model",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            evaluation_strategy="steps",
            eval_steps=1,
            logging_steps=1,
            gradient_accumulation_steps=1,
            num_train_epochs=2,
            weight_decay=0.1,
            lr_scheduler_type="cosine",
            learning_rate=1e-4,
            save_steps=5_000,
            generation_max_length=512,
            predict_with_generate=True,  # generation in evaluation
            prediction_loss_only=False,  # generation in evaluation
        ),
        save_model_dir: str = "flant5-model",
        print_prediction_num_examples: int = 300,
    ) -> Model:
        """
        Train the explain model on the given dataset.

        Args:
            dataset (Dataset): The dataset to train on. A single example is a dictionary with the following keys:
            - input_prompt
            - output_prompt

        Returns:
            Explain model: The trained explain model.
        """
        prompt_train_dataset = dataset["train"].map(
            partial(map_dataset_to_tokenized_prompt, self.tokenizer),
            batched=True,
            load_from_cache_file=False,
            remove_columns=dataset["train"].column_names,
        )
        
        print("-"*50)
        print(f"Snippet of training dataset:")
        print(prompt_train_dataset)
        print("-"*50)
        
        
        prompt_train_dataset = cast(TorchDataset, prompt_train_dataset)  # type: ignore
        prompt_valid_dataset = dataset["validation"].map(
            partial(map_dataset_to_tokenized_prompt, self.tokenizer),
            batched=True,
            load_from_cache_file=False,
            remove_columns=dataset["validation"].column_names,
        )
        prompt_valid_dataset = cast(TorchDataset, prompt_valid_dataset)  # type: ignore

        self.tokenizer.pad_token = self.tokenizer.eos_token
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)

        trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=prompt_train_dataset,
            eval_dataset=prompt_valid_dataset,
            compute_metrics=partial(
                aggregated_metrics_with_postprocess,
                [
                    partial(prediction_metrics, print_prediction_num_examples),
                    bleu_metrics,
                ],
                self.tokenizer,
            ),
        )
        trainer.train()

        if save_model_dir != "":
            trainer.save_model(save_model_dir)

        return self

    def predict(
        self,
        dataset: Dataset,
        args: Seq2SeqTrainingArguments = Seq2SeqTrainingArguments(
            output_dir="checkpoints/_model",
            per_device_eval_batch_size=16,
            generation_max_length=512,
            generation_num_beams=4,
            predict_with_generate=True,  # generation in evaluation
            prediction_loss_only=False,  # generation in evaluation
        ),
        gen_kwargs: Dict[str, Any] = {},
    ) -> Dict[str, List[str]]:
        """
        Predict the reward for the given dataset.

        Args:
            dataset (Dataset): The dataset to predict on. A single example is a dictionary with the same keys as train above except for "labels".

        Returns:
            Dict[str, List[str]]: The predicted explanation for each example in the dataset, with the following keys:
                -
        """
        prompt_dataset = dataset.map(
            partial(
                map_dataset_to_tokenized_prompt,
                self.tokenizer,
            ),
            batched=True,
            load_from_cache_file=False,
            remove_columns=dataset.column_names,
        )
        
        
        prompt_dataset = cast(TorchDataset, prompt_dataset)  # type: ignore
        # limit dataset size for faster evaluation
        data_collator = DataCollatorForSeq2Seq(self.tokenizer)

        self.model.to('cuda')
        trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=None,
            eval_dataset=prompt_dataset,
        )

        predict_results = trainer.predict(
            prompt_dataset,
            metric_key_prefix="predict",
            **gen_kwargs,
        )
        predictions = predict_results.predictions
        # print(predictions)
        # print(predictions.shape)
        predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
        predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )       
                
        # print("-"*50)
        # print(f"Predictions:")
        # print(predictions)
        # print("-"*50)
        
        answer_dict: Dict[str, List[str]] = {'prediction': []}
        for txt in predictions:
            answer_dict["prediction"].append(txt.strip())

        return answer_dict
