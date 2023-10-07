import json
import os
import sys
import pandas as pd
from pprint import pprint
from tqdm import tqdm
tqdm.pandas()

from prompts import DEPRESSION_FEWSHOT_LANGCHAIN, ANXIETY_FEWSHOT_LANGCHAIN, COMORBIDITY_FEWSHOT_LANGCHAIN

text_col = 'text'
id_col = 'id'

sys.path.insert(0,'')
test_path = 'data/test/full_test.csv'
corpus_path = 'data/silver_data/silver_labels_gpt_3.5_turbo.csv'
mapping_path_depression = "data/mappings/semantic-similarity-depression.json"
mapping_path_anxiety = "data/mappings/semantic-similarity-anxiety.json"
mapping_path_comorbid = "data/mappings/semantic-similarity-comorbid.json"
mapping_path_normal = "data/mappings/semantic-similarity-normal.json"


df_test = pd.read_csv(test_path).reset_index()
df_corpus = pd.read_csv(corpus_path).reset_index()
mapping_depression = json.load(open(mapping_path_depression))
mapping_anxiety = json.load(open(mapping_path_anxiety))
mapping_comorbid = json.load(open(mapping_path_comorbid))
mapping_normal = json.load(open(mapping_path_normal))


print('-'*50)
print(f"Test set size: {df_test.shape[0]}")
print(f"Corpus (Silver Labels) size: {df_corpus.shape[0]}")
print(f"Total mapping (Depression): {len(mapping_depression)}")
print(f"Total mapping (Anxiety): {len(mapping_anxiety)}")
print(f"Total mapping (Comorbid): {len(mapping_comorbid)}")
print(f"Total mapping (Normal): {len(mapping_normal)}")
print('-'*50)


def merge_dicts(dict1, dict2):
    merged_dict = {}
    for i in dict1.keys():
        merged_dict[i] = dict1[i]
    for i in dict2.keys():
        merged_dict[i] = dict2[i]
    return merged_dict

df_corpus['depression_label'] = df_corpus['depression_label'].apply(lambda label: {"depression": "yes"} if label == 1 else {"depression": "no"})
df_corpus['anxiety_label'] = df_corpus['anxiety_label'].apply(lambda label: {"anxiety": "yes"} if label == 1 else {"anxiety": "no"})
df_corpus['comorbidity_label'] = df_corpus.apply(lambda row: merge_dicts(row['depression_label'],row['anxiety_label']), axis=1)


def generate_few_shot_prompts(topk=4, task_type='depression'):
    prompts = []
    
    for i in tqdm(range(df_test.shape[0]), desc=f"Generating few shot prompts"):
        few_shot_examples = []
        input_post = df_test.iloc[i][text_col]
        
        if task_type == 'depression':        
            exemplars_d = mapping_depression[df_test.iloc[i][id_col]]
            exemplar_ids_d = [i[0] for i in exemplars_d][:int(topk/2)]
            exemplars_n = mapping_normal[df_test.iloc[i][id_col]]
            exemplar_ids_n = [i[0] for i in exemplars_n][:int(topk/2)]
            exemplar_ids = exemplar_ids_d + exemplar_ids_n
    
            exemplar_df = df_corpus[df_corpus[id_col].isin(exemplar_ids)]
            exemplar_df = exemplar_df.sample(exemplar_df.shape[0])
            
            for j, row in exemplar_df.iterrows():
                exemplar_post = row['text']
                exemplar_label = row['depression_label']
                
                few_shot_examples.append(
                    {
                        'post': exemplar_post,
                        'label': exemplar_label,
                    }
                )
                
            prompt_template = DEPRESSION_FEWSHOT_LANGCHAIN['prompt_template']
            few_shot_prefix = DEPRESSION_FEWSHOT_LANGCHAIN['few_shot_prefix']
            few_shot_suffix = DEPRESSION_FEWSHOT_LANGCHAIN['few_shot_suffix'](input_post)
            
        
        elif task_type == 'anxiety':
            exemplars_a = mapping_anxiety[df_test.iloc[i][id_col]]
            exemplar_ids_a = [i[0] for i in exemplars_a][:int(topk/2)]
            exemplars_n = mapping_normal[df_test.iloc[i][id_col]]
            exemplar_ids_n = [i[0] for i in exemplars_n][:int(topk/2)]
            exemplar_ids = exemplar_ids_a + exemplar_ids_n

            exemplar_df = df_corpus[df_corpus[id_col].isin(exemplar_ids)]
            exemplar_df = exemplar_df.sample(exemplar_df.shape[0])

            for j, row in exemplar_df.iterrows():
                exemplar_post = row['text']
                exemplar_label = row['anxiety_label']
                
                few_shot_examples.append(
                    {
                        'post': exemplar_post,
                        'label': exemplar_label,
                    }
                )
                
            prompt_template = ANXIETY_FEWSHOT_LANGCHAIN['prompt_template']
            few_shot_prefix = ANXIETY_FEWSHOT_LANGCHAIN['few_shot_prefix']
            few_shot_suffix = ANXIETY_FEWSHOT_LANGCHAIN['few_shot_suffix'](input_post)
            
            
        elif task_type == 'comorbidity':
            if topk < 4: 
                topk = 4

            exemplars_d = mapping_depression[df_test.iloc[i][id_col]]
            exemplar_ids_d = [i[0] for i in exemplars_d][:int(topk/4)]
            exemplars_a = mapping_anxiety[df_test.iloc[i][id_col]]
            exemplar_ids_a = [i[0] for i in exemplars_a][:int(topk/4)]
            exemplars_c = mapping_comorbid[df_test.iloc[i][id_col]]
            exemplar_ids_c = [i[0] for i in exemplars_c][:int(topk/4)]
            exemplars_n = mapping_normal[df_test.iloc[i][id_col]]
            exemplar_ids_n = [i[0] for i in exemplars_n][:int(topk/4)]
            exemplar_ids = exemplar_ids_d + exemplar_ids_a + exemplar_ids_c + exemplar_ids_n

            exemplar_df = df_corpus[df_corpus[id_col].isin(exemplar_ids)]
            exemplar_df = exemplar_df.sample(exemplar_df.shape[0])

            for j, row in exemplar_df.iterrows():
                exemplar_post = row['text']
                exemplar_label = row['comorbidity_label']
                
                few_shot_examples.append(
                    {
                        'post': exemplar_post,
                        'label': exemplar_label,
                    }
                )
                
            prompt_template = COMORBIDITY_FEWSHOT_LANGCHAIN['prompt_template']
            few_shot_prefix = COMORBIDITY_FEWSHOT_LANGCHAIN['few_shot_prefix']
            few_shot_suffix = COMORBIDITY_FEWSHOT_LANGCHAIN['few_shot_suffix'](input_post)
                
            
        few_shot_examples = ''.join(prompt_template(x['post'], x['label']) for x in few_shot_examples)
        few_shot_prompt = ''.join([few_shot_prefix, few_shot_examples, few_shot_suffix])        
        prompts.append(few_shot_prompt)

    return prompts


import argparse

def main(task_type, topk, outfile):
    print('-'*50)
    print(f"Task Type: {task_type}")
    print(f"Top K: {topk}")
    print(f"Output File: {outfile}")
    print('-'*50)
    
    df_test[f'few_shot_prompt_{task_type}'] = generate_few_shot_prompts(topk, task_type)
    df_test.to_csv(outfile, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument("task_type", type=str, default="depression" , help="(depression | anxiety | comorbidity)")
    parser.add_argument("topk", type=int, default=4, help="Top K value.")
    parser.add_argument("outfile", type=str, help="Path to output file.")

    args = parser.parse_args()
    main(args.task_type, args.topk, args.outfile)
    
    
"""
Usage:
python src/few-shot-prompting.py \
    comorbidity \
    8 \
    data/few_shot_prompts/comorbidity_8_ise.csv \
    ;
"""