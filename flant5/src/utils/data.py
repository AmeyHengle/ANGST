import datasets
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
import pandas as pd
import sys
import json

DEPRESSION_ZEROSHOT = lambda post: f'''
Given the below post, assess whether its content aligns with the criteria for clinical depression as defined in the DSM-5.
Format your response as an object ('depression':'') with values either 'yes' or 'no'.

Post: {post}
'''.replace('{','(').replace('}',')').strip()

ANXIETY_ZEROSHOT = lambda post: f'''
Given the below post, assess whether its content aligns with the criteria for clinical anxiety as defined in the DSM-5.
Format your response as a JSON object ('anxiety':'') with values either 'yes' or 'no'.

Post: {post}
'''.replace('{','(').replace('}',')').strip()

COMORBIDITY_ZEROSHOT = lambda post: f'''
Given the below post, assess whether its content aligns with the criteria for clinical anxiety and clinical depression respectively as defined in the DSM-5.
Format your response as a JSON object ('depression':'', anxiety: '') with values either 'yes' or 'no'.

Post: {post}
'''.replace('{','(').replace('}',')').strip()

OUTPUT_PROMPT = lambda label: f'''
Assessment: {label}
'''.replace('{','(').replace('}',')').strip()

def get_data(mode: str, split: str = ""):
    if mode == "train":
        return process_data(mode, split=split)
    elif mode == "test":
        return process_data(mode, split=split)
    else:
        raise ValueError(f"Unknown mode {mode}")


def merge_dicts(dict1, dict2):
    merged_dict = {}
    for i in dict1.keys():
        merged_dict[i] = dict1[i]
    for i in dict2.keys():
        merged_dict[i] = dict2[i]
    return merged_dict


def preprocess(df, split):
    df['depression_label'] = df['depression_label'].apply(lambda label: {"depression": "yes"} if label == 1 else {"depression": "no"})
    df['anxiety_label'] = df['anxiety_label'].apply(lambda label: {"anxiety": "yes"} if label == 1 else {"anxiety": "no"})
    df['comorbidity_label'] = df.apply(lambda row: merge_dicts(row['depression_label'], row['anxiety_label']), axis=1)
            
    if split == 'depression':
        df['input_prompt'] = df['text'].apply(lambda x: DEPRESSION_ZEROSHOT(x))
        df['output_prompt'] = df['depression_label'].apply(lambda x: OUTPUT_PROMPT(x))
    elif split == 'anxiety':
        df['input_prompt'] = df['text'].apply(lambda x: ANXIETY_ZEROSHOT(x))
        df['output_prompt'] = df['anxiety_label'].apply(lambda x: OUTPUT_PROMPT(x))
    elif split == 'comorbidity':
        df['input_prompt'] = df['text'].apply(lambda x: COMORBIDITY_ZEROSHOT(x))
        df['output_prompt'] = df['comorbidity_label'].apply(lambda x: OUTPUT_PROMPT(x))
        
    return df


def process_data(mode, split: str = "") -> DatasetDict:
    if mode=='train':
        df_train = pd.read_csv('../data/silver_data/silver_labels_gpt_3.5_turbo_train.csv')
        df_val = pd.read_csv('../data/silver_data/silver_labels_gpt_3.5_turbo_validation.csv')
        
        df_train = preprocess(df_train, split)
        df_val = preprocess(df_val, split)
        
        train_dataset = Dataset.from_pandas(df_train)
        val_dataset = Dataset.from_pandas(df_val)
        
        print(f"train_dataset: {len(train_dataset)}")
        print(f"val_dataset: {len(val_dataset)}")

        ds = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })      
        return ds # type: ignore
    
    elif mode == "test":
        df_test = pd.read_csv('../data/test/full_test.csv').sample(10)
        df_test = preprocess(df_test, split)
        ds = Dataset.from_pandas(df_test)
        return ds # type: ignore
    
    else:
        raise Exception("Invalid split, must be one of 'train' or 'test'")