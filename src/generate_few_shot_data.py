import argparse
import json
import random
import pandas as pd
from tqdm import tqdm

from prompts import (
DEPRESSION_FEWSHOT_LANGCHAIN,
ANXIETY_FEWSHOT_LANGCHAIN,
COMORBIDITY_FEWSHOT_LANGCHAIN,
DEPRESSION_FEWSHOT_LLAMA_LANGCHAIN,
ANXIETY_FEWSHOT_LLAMA_LANGCHAIN,
COMORBIDITY_FEWSHOT_LLAMA_LANGCHAIN
)

text_col = 'text'
id_col = 'id'

test_path = './data/test/full_test.csv'
corpus_path = './data/silver_data/silver_labels_gpt_3.5_turbo.csv'
mapping_path_depression = "./data/mappings/semantic-similarity-depression.json"
mapping_path_anxiety = "./data/mappings/semantic-similarity-anxiety.json"
mapping_path_comorbid = "./data/mappings/semantic-similarity-comorbid.json"
mapping_path_normal = "./data/mappings/semantic-similarity-normal.json"
mapping_path_semantic_similiary = "./data/mappings/semantic-similarity-only.json"

df_test = pd.read_csv(test_path).reset_index()
df_corpus = pd.read_csv(corpus_path).reset_index()
mapping_depression = json.load(open(mapping_path_depression))
mapping_anxiety = json.load(open(mapping_path_anxiety))
mapping_comorbid = json.load(open(mapping_path_comorbid))
mapping_normal = json.load(open(mapping_path_normal))
mapping_semantic_similiary = json.load(open(mapping_path_semantic_similiary))

print('-'*50)
print(f"Test set size: {df_test.shape[0]}")
print(f"Corpus (Silver Labels) size: {df_corpus.shape[0]}")
print(f"Total mapping (Depression): {len(mapping_depression)}")
print(f"Total mapping (Anxiety): {len(mapping_anxiety)}")
print(f"Total mapping (Comorbid): {len(mapping_comorbid)}")
print(f"Total mapping (Normal): {len(mapping_normal)}")
print(f"Total mapping (Semantic Similarity): {len(mapping_semantic_similiary)}")
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



def generate_few_shot_prompts(
    data_type='depression', num_icl_examples=4, use_semantic_similarity_only=False, model="gpt-3.5-turbo", outfile="", icl_template=None
):
    prompts = []
    exemplar_labels = []
    
    for i in tqdm(range(df_test.shape[0]), desc=f"Generating few shot prompts"):
        few_shot_examples = []
        few_shot_labels = []
        input_post = df_test.iloc[i][text_col]
        
        if data_type == 'depression':
            if use_semantic_similarity_only:
                exemplars = mapping_semantic_similiary[df_test.iloc[i][id_col]]
                exemplar_ids = [i[0] for i in exemplars][:int(num_icl_examples)]

            else:
                exemplars_d = mapping_depression[df_test.iloc[i][id_col]]
                exemplar_ids_d = [i[0] for i in exemplars_d][:int(num_icl_examples/2)]
                exemplars_n = mapping_normal[df_test.iloc[i][id_col]]
                exemplar_ids_n = [i[0] for i in exemplars_n][:int(num_icl_examples/2)]
                exemplar_ids = exemplar_ids_d + exemplar_ids_n
    
            exemplar_df = df_corpus[df_corpus[id_col].isin(exemplar_ids)]
            exemplar_df = exemplar_df.sample(exemplar_df.shape[0])
            
            for j, row in exemplar_df.iterrows():
                exemplar_post = row['text']
                
                if "llama" in model:
                    exemplar_label = row['depression_label']['depression']
                    # exemplar_label = {
                    #     "yes": "Yes, the poster suffers from depression. Reasoning: The language of the post and the symptoms mentioned are indicative of depression as per DSM-5.",
                    #     "no": "No, the poster does not suffer from depression. Reasoning: The language of the post and the symptoms mentioned are NOT indicative of depression as per DSM-5."
                    # }[exemplar_label.lower()]
                    exemplar_label = {
                        "yes": "Yes, the poster suffers from depression.",
                        "no": "No, the poster does not suffer from depression."
                    }[exemplar_label.lower()]
                
                else:
                    exemplar_label = row['depression_label']
                    
                few_shot_labels.append(exemplar_label)
                few_shot_examples.append(
                    {
                        'post': exemplar_post,
                        'label': exemplar_label,
                    }
                )
      
        elif data_type == 'anxiety':
            if use_semantic_similarity_only:
                exemplars = mapping_semantic_similiary[df_test.iloc[i][id_col]]
                exemplar_ids = [i[0] for i in exemplars][:int(num_icl_examples)]

            else:
                exemplars_a = mapping_anxiety[df_test.iloc[i][id_col]]
                exemplar_ids_a = [i[0] for i in exemplars_a][:int(num_icl_examples/2)]
                exemplars_n = mapping_normal[df_test.iloc[i][id_col]]
                exemplar_ids_n = [i[0] for i in exemplars_n][:int(num_icl_examples/2)]
                exemplar_ids = exemplar_ids_a + exemplar_ids_n

            exemplar_df = df_corpus[df_corpus[id_col].isin(exemplar_ids)]
            exemplar_df = exemplar_df.sample(exemplar_df.shape[0])

            for j, row in exemplar_df.iterrows():
                exemplar_post = row['text']
                
                if "llama" in model:
                    exemplar_label = row['anxiety_label']['anxiety']
                    exemplar_label = {
                        "yes": "Yes, the poster suffers from anxiety. Reasoning: The language of the post and the symptoms mentioned are indicative of anxiety as per DSM-5.",
                        "no": "No, the poster does not suffer from anxiety. Reasoning: The language of the post and the symptoms mentioned are NOT indicative of anxiety as per DSM-5."
                    }[exemplar_label.lower()]
                else:
                    exemplar_label = row['anxiety_label']
                    
                few_shot_labels.append(exemplar_label)
                few_shot_examples.append(
                    {
                        'post': exemplar_post,
                        'label': exemplar_label,
                    }
                )

        elif data_type == 'comorbidity':
            if num_icl_examples < 4: 
                num_icl_examples = 4

            if use_semantic_similarity_only:
                exemplars = mapping_semantic_similiary[df_test.iloc[i][id_col]]
                exemplar_ids = [i[0] for i in exemplars][:int(num_icl_examples)]

            else:
                exemplars_d = mapping_depression[df_test.iloc[i][id_col]]
                exemplar_ids_d = [i[0] for i in exemplars_d][:int(num_icl_examples/4)]
                exemplars_a = mapping_anxiety[df_test.iloc[i][id_col]]
                exemplar_ids_a = [i[0] for i in exemplars_a][:int(num_icl_examples/4)]
                exemplars_c = mapping_comorbid[df_test.iloc[i][id_col]]
                exemplar_ids_c = [i[0] for i in exemplars_c][:int(num_icl_examples/4)]
                exemplars_n = mapping_normal[df_test.iloc[i][id_col]]
                exemplar_ids_n = [i[0] for i in exemplars_n][:int(num_icl_examples/4)]
                exemplar_ids = exemplar_ids_d + exemplar_ids_a + exemplar_ids_c + exemplar_ids_n

            exemplar_df = df_corpus[df_corpus[id_col].isin(exemplar_ids)]
            exemplar_df = exemplar_df.sample(exemplar_df.shape[0])

            for j, row in exemplar_df.iterrows():
                exemplar_post = row['text']
                if "llama" in model:
                    exemplar_label = " and ".join([f"{key} {value}" for key, value in row['comorbidity_label'].items()])
                else:
                    exemplar_label = row['comorbidity_label']
                    
                few_shot_labels.append(exemplar_label)
                few_shot_examples.append(
                    {
                        'post': exemplar_post,
                        'label': exemplar_label,
                    }
                )

        prompt_template = icl_template['prompt_template']
        few_shot_prefix = icl_template['few_shot_prefix']
        few_shot_suffix = icl_template['few_shot_suffix'](input_post)
        
        few_shot_examples = ''.join(prompt_template(x['post'], x['label']) for x in few_shot_examples)
        few_shot_prompt = ''.join([few_shot_prefix, few_shot_examples, few_shot_suffix])     
        prompts.append(few_shot_prompt)
        exemplar_labels.append(few_shot_labels)
        
    return prompts, exemplar_labels



def main(
    data_type, num_icl_examples, use_semantic_similarity_only, model, outfile
):
    print('-'*50)
    print(f"Task Type: {data_type}")
    print(f"Top K: {num_icl_examples}")
    print(f"Output File: {outfile}")
    print('-'*50)
    
    if args.data_type == "depression":
        if "llama" in args.model:
            icl_icl_template = DEPRESSION_FEWSHOT_LLAMA_LANGCHAIN
        else:
            icl_icl_template = DEPRESSION_FEWSHOT_LANGCHAIN
            
    elif args.data_type == "anxiety":
        if "llama" in args.model:
            icl_icl_template = ANXIETY_FEWSHOT_LLAMA_LANGCHAIN
        else:
            icl_icl_template = ANXIETY_FEWSHOT_LANGCHAIN
            
    elif args.data_type == "comorbidity":
        if "llama" in args.model:
            icl_icl_template = COMORBIDITY_FEWSHOT_LLAMA_LANGCHAIN
        else:
            icl_icl_template = COMORBIDITY_FEWSHOT_LANGCHAIN
    
    prompts, exemplar_labels = generate_few_shot_prompts(data_type, num_icl_examples, use_semantic_similarity_only, model, outfile, icl_icl_template)
    df_test[f'few_shot_prompt_{data_type}'] = prompts
    df_test[f'exemplar_labels_{data_type}'] = exemplar_labels
    
    index = random.randint(0, len(df_test))
    print(f"\nSample Input: {df_test.at[index, f'few_shot_prompt_{data_type}']}\n")
         
    df_test.to_csv(outfile, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line arguments.")
    parser.add_argument(
        "--data_type", 
        type=str, 
        default="depression" , 
        choices=['depression', 'anxiety', 'comorbidity'],
        help="The data for which to retrieve few-shot examples"
    )
    parser.add_argument(
        "--num_icl_examples", 
        type=int, 
        default=4, 
        help="The number of examples to be included for in-context learning"
    )
    parser.add_argument(
        "--use_semantic_similarity_only", 
        action="store_true", 
        help="To rerieve examples only based on semantic similarity"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4", "mental_llama_chat_7b", "mental_llama_chat_13b"],
        help="type of model to use for prompting later."
    )
    parser.add_argument(
        "--outfile", 
        type=str, 
        help="Path to output file."
    )
    args = parser.parse_args()
    
    main(args.data_type, args.num_icl_examples, args.use_semantic_similarity_only, args.model, args.outfile)