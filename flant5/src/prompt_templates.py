from typing import Dict, List, cast

import gin
import numpy as np
from datasets.arrow_dataset import Dataset
from transformers import PreTrainedTokenizer

CONTEXT_LENGTH = 1024
TARGET_LENGTH = 64

@gin.configurable
def input_prompt_template_base(
    with_context: bool = True,
):
    return "{input_prompt}".strip()

@gin.configurable
def output_prompt_template_base(
    with_context: bool = True,
):
    return "{output_prompt}".strip()


@gin.configurable
def input_prompt_template_depression(
    with_context: bool = True,
):
    input_prompt = """
Given the below post, assess whether its content aligns with the criteria for clinical depression as defined in the DSM-5.
Format your response as an object ('depression':'') with values either 'yes' or 'no'.

Post: {text}
""".strip()
    return input_prompt

@gin.configurable
def output_prompt_template_depression(
    with_context: bool = True,
):
    output_prompt = """
Assessment: {prompt_label}
""".strip()
    return output_prompt


@gin.configurable
def input_prompt_template_anxiety(
    with_context: bool = True,
):
    input_prompt = """
Given the below post, assess whether its content aligns with the criteria for clinical anxiety as defined in the DSM-5.
Format your response as an object ('anxiety':'') with values either 'yes' or 'no'.

Post: {text}
""".strip()
    return input_prompt

@gin.configurable
def output_prompt_template_anxiety(
    with_context: bool = True,
):
    output_prompt = """
Assessment: {prompt_label}
""".strip()
    return output_prompt


@gin.configurable
def input_prompt_template_comorbidity(
    with_context: bool = True,
):
    input_prompt = """
Given the below post, assess whether its content aligns with the criteria for clinical anxiety as defined in the DSM-5.
Format your response as an object ('depression':'', 'anxiety':'') with values either 'yes' or 'no'.

Post: {text}
""".strip()
    return input_prompt

@gin.configurable
def output_prompt_template_comorbidity(
    with_context: bool = True,
):
    output_prompt = """
Assessment: {prompt_label}
""".strip()
    return output_prompt


def map_dataset_to_tokenized_prompt(
    tokenizer: PreTrainedTokenizer, element: Dataset, with_labels: bool = True
) -> Dict[str, np.ndarray]:
    element_cast: Dict[str, List[str]] = cast(Dict[str, List[str]], element)  # type: ignore
    context_input = map(
        lambda instance: input_prompt_template_base().format(**instance),
        (dict(zip(element_cast, t)) for t in zip(*element_cast.values())),
    )
    context_input_ = map(
        lambda instance: input_prompt_template_base().format(**instance),
        (dict(zip(element_cast, t)) for t in zip(*element_cast.values())),
    )
    
    print('-'*50)
    print(f"Context Input:")
    print(list(context_input_)[0])
    print('-'*50)
        
    tokenized_context_input = tokenizer(
        list(context_input),
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_overflowing_tokens=True,
        return_length=True,
    )
    if with_labels:
        target = map(
            lambda instance: output_prompt_template_base().format(**instance),
            (dict(zip(element_cast, t)) for t in zip(*element_cast.values())),
        )
        target_ = map(
            lambda instance: output_prompt_template_base().format(**instance),
            (dict(zip(element_cast, t)) for t in zip(*element_cast.values())),
        )
        print('-'*50)
        print(f"Target:")
        print(list(target_)[0])
        print('-'*50)

        tokenized_target = tokenizer(
            list(target),
            truncation=True,
            max_length=TARGET_LENGTH,
            return_overflowing_tokens=True,
            return_length=True,
        )
        tokenized_context_input.update(
            {"labels": tokenized_target["input_ids"]}
        )

    return tokenized_context_input  # type: ignore