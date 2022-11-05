"""
python silver_labelling.py \
    --train_file=data/train/v0.0.0.csv \
    --corpus_file=data/corpus/unlabeled_corpus.csv \
    --outfile=data/train/v0.1.0.csv \
    --mapping_ss=data/mappings/semantic-similarity-example.json \
    ;
"""

import pandas as pd
import numpy as np
import json
import argparse
import constants as const
from loguru import logger
from collections import Counter
from utils import clean_text, label_to_word, word_to_label, load_json
from prompting import (
    gpt,
    prompt_template1,
    prompt_template2,
    header1
)
from tqdm import tqdm
tqdm.pandas()

"""
Algorithm:
Input: data point (id) from df_unlabeled, df_train, mapping_dict, silver_label_technique
Output: df_unlabeled populated with [silver_labels]
"""
def get_silver_label(
    post: str,
    id: str, 
    search_df: pd.DataFrame, 
    mapping_ss: dict,
    mapping_dm25: dict,
    threshold_ss: float,
    threshold_dm25: float,
    topk: int, 
    id_col: str='id',
    label_col: str='labels',
    text_col: str='selftext',
    silver_label_technique: str='ss',
    use_gpt: bool=False,
    prompt_header=None,
    prompt_template=None
    ):
    silver_label = None
    matches = pd.DataFrame()

    # --------------------------------------------------------------------------
    # Derive topk most-similar data points from train set
    # --------------------------------------------------------------------------
    if silver_label_technique == 'ss':
        if not mapping_ss:
            raise Exception(
                "missing mapping_ss"    
            )
        search_keys = [x[0] for x in mapping_ss[id] if x[1] >= threshold_ss][:topk]
        #print(search_keys)
        matches = search_df[search_df[id_col].isin(search_keys)]
        
    elif silver_label_technique == 'bm25':
        if not mapping_dm25:
            raise Exception(
                "missing mapping_dm25"
            )
        search_keys = [x[0] for x in mapping_ss[id] if x[1] >= mapping_dm25][:topk]
        #print(search_keys)
        matches = search_df[search_df[id_col].isin(search_keys)]
       
    elif silver_label_technique == 'ss+bm25':
        if not mapping_ss or not mapping_dm25:
            raise Exception(
                f'Missing mapping_ss or mapping_dm25'
            )
        search_keys1 = [x[0] for x in mapping_ss[id] if x[1] >= threshold_ss][:topk]
        search_keys2 = [x[0] for x in mapping_ss[id] if x[1] >= threshold_dm25][:topk]
        search_keys = list(set(search_keys1).intersection(search_keys2))
        #print(search_keys)
        matches = search_df[search_df[id_col].isin(search_keys)]

    # --------------------------------------------------------------------------
    # Generate Silver Label
    # --------------------------------------------------------------------------    
    if matches.shape[0] > 0:
        # --------------------------------------------------------------------------
        # Silver Label using GPT
        # --------------------------------------------------------------------------           
        if use_gpt:
            examples = matches[text_col].values.tolist()
            groud_truth_labels = matches[label_col].values.tolist()
            
            prompt = prompt_header
            for text, label in zip(examples,groud_truth_labels):
                disorder = label_to_word(label)
                prompt += prompt_template(text, disorder)
            prompt += prompt_template(post,'')
            
            # #print(prompt)
            gpt_pred = gpt(prompt)
                                    
            if gpt_pred:
                silver_label = word_to_label(gpt_pred)
                
        # --------------------------------------------------------------------------
        # Silver Label without GPT (Assigning Label of best match)
        # -------------------------------------------------------------------------- 
        else:
            groud_truth_labels = matches[label_col].values.tolist()
            best_match = Counter(groud_truth_labels).most_common(1)[0][0]
            if best_match != None:
                silver_label = best_match

    
    return silver_label


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file")
    parser.add_argument("--corpus_file")
    parser.add_argument("--outfile")
    parser.add_argument("--mapping_ss",default={})
    parser.add_argument("--mapping_dm25",default={})
    parser.add_argument("--config", default=const.CONFIG_SILVER_LABELLING)
    args = parser.parse_args()

    datapath_train = args.train_file
    datapath_corpus = args.corpus_file
    outfile = args.outfile
    config = args.config
    mapping_ss = args.mapping_ss
    mapping_dm25 = args.mapping_dm25
    
    df_train = pd.read_csv(datapath_train)
    df_corpus = pd.read_csv(datapath_corpus)


    # Initialize variables
    id_col = const.ID_COL
    text_col = const.TEXT_COL
    silver_label_technique = config['technique'] or 'ss'
    use_gpt = config['use_gpt']
    topk = config['topk']
    threshold_ss = config['threshold_ss']
    threshold_dm25 = config['threshold_dm25']
    
    mapping_ss = load_json(mapping_ss) if mapping_ss!={} else mapping_ss
    mapping_dm25 = load_json(mapping_dm25) if mapping_dm25!={} else mapping_dm25

    prompt_header = header1
    prompt_template = prompt_template2
    
    
    # Fill label column if df_corpus with silver_labeld
    if 'level_0' in df_corpus.columns:
        df_corpus = df_corpus.drop(['level_0'],axis=1)
    df_corpus = df_corpus.reset_index()
    df_corpus[silver_label_technique] = None
    
    for i in tqdm(range(df_corpus.shape[0]),desc=f"Generating silver labels"):
        post = df_corpus.iloc[i][text_col]
        id = df_corpus.iloc[i][id_col]
        silver_label =  get_silver_label(
                            post,
                            id,
                            search_df=df_train, 
                            text_col=text_col,
                            id_col=id_col,
                            mapping_ss=mapping_ss,
                            mapping_dm25=mapping_dm25,
                            threshold_ss=threshold_ss,
                            threshold_dm25=threshold_dm25,
                            topk=topk,
                            silver_label_technique=silver_label_technique,
                            use_gpt=use_gpt,
                            prompt_header=prompt_header,
                            prompt_template=prompt_template
                        )
        df_corpus.at[i,silver_label_technique] = silver_label
        
    
    # Save df_corpus populated with silver_labels
    logger.debug(f"Silver label dist:\n{df_corpus[silver_label_technique].value_counts()}")
    df_corpus.to_csv(outfile, index=False)