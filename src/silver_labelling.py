"""
python silver_labelling.py \
    --train_file=data/train/v0.0.0.csv \
    --corpus_file=data/corpus/unlabeled_corpus.csv \
    --outfile=data/train/v1.0.0.csv \
    --mapping_bm25=data/mappings/bm25.json \
    --mapping_ss=data/mappings/semantic-similarity.json \
    ;
"""

import pandas as pd
import numpy as np
import json
import argparse
import constants as const
from loguru import logger
from collections import Counter
from utils import clean_text, label_to_word, word_to_label, load_json, set_random_seed
from prompting import (
    gpt,
    bloom,
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
    mapping_bm25: dict,
    threshold_ss: float,
    threshold_bm25: float,
    topk: int, 
    id_col: str='id',
    label_col: str='labels',
    text_col: str='selftext',
    silver_label_technique: str='ss',
    use_nlg: bool=False,
    prompt_header=None,
    prompt_template=None,
    nlg_pipeline=None
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
        if not mapping_bm25:
            raise Exception(
                "missing mapping_bm25"
            )
        search_keys = [x[0] for x in mapping_bm25[id]][:topk]
        #print(search_keys)
        matches = search_df[search_df[id_col].isin(search_keys)]
       
    elif silver_label_technique == 'ss+bm25':
        if not mapping_ss or not mapping_bm25:
            raise Exception(
                f'Missing mapping_ss or mapping_bm25'
            )
        search_keys1 = [x[0] for x in mapping_ss[id] if x[1] >= threshold_ss][:topk]
        search_keys2 = [x[0] for x in mapping_bm25[id]][:topk]
        search_keys = list(set(search_keys1).intersection(search_keys2))
        #print(search_keys)
        matches = search_df[search_df[id_col].isin(search_keys)]

    # --------------------------------------------------------------------------
    # Generate Silver Label
    # --------------------------------------------------------------------------    
    if matches.shape[0] > 0:
        # --------------------------------------------------------------------------
        # Silver Label using GPT and BLOOM
        # --------------------------------------------------------------------------           
        if use_nlg:
            examples = matches[text_col].values.tolist()
            groud_truth_labels = matches[label_col].values.tolist()
            
            prompt = prompt_header
            for text, label in zip(examples,groud_truth_labels):
                disorder = label_to_word(label)
                prompt += prompt_template(text, disorder)
            prompt += prompt_template(post,'')

            # print(prompt)
            
            if nlg_pipeline == 'bloom':
                text_generated = bloom(prompt)
                logger.debug(f"bloom prediction: {text_generated}")
            else:
                text_generated = gpt(prompt)
                logger.debug(f"gpt prediction: {text_generated}")
                                    
            if text_generated:
                silver_label = word_to_label(text_generated)
                
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
    SEED = const.RANDOM_STATE
    set_random_seed(SEED)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file")
    parser.add_argument("--corpus_file")
    parser.add_argument("--outfile")
    parser.add_argument("--mapping_ss",default={})
    parser.add_argument("--mapping_bm25",default={})
    parser.add_argument("--config", default=const.CONFIG_SILVER_LABELLING)
    parser.add_argument("--concat_with_train", default=True)
    args = parser.parse_args()

    datapath_train = args.train_file
    datapath_corpus = args.corpus_file
    outfile = args.outfile
    config = args.config
    mapping_ss = args.mapping_ss
    mapping_bm25 = args.mapping_bm25
    
    df_train = pd.read_csv(datapath_train)
    df_corpus = pd.read_csv(datapath_corpus)

    # Initialize variables
    id_col = const.ID_COL
    text_col = const.TEXT_COL
    silver_label_technique = config['technique'] or 'ss'
    use_nlg = config['use_nlg']
    topk = config['topk']
    threshold_ss = config['threshold_ss']
    threshold_bm25 = config['threshold_bm25']
    nlg_pipeline = config['nlg_pipeline'] or 'gpt'
    
    mapping_ss = load_json(mapping_ss) if mapping_ss!={} else mapping_ss
    mapping_bm25 = load_json(mapping_bm25) if mapping_bm25!={} else mapping_bm25

    prompt_header = header1
    prompt_template = prompt_template1
    
    
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
                            mapping_bm25=mapping_bm25,
                            threshold_ss=threshold_ss,
                            threshold_bm25=threshold_bm25,
                            topk=topk,
                            silver_label_technique=silver_label_technique,
                            use_nlg=use_nlg,
                            prompt_header=prompt_header,
                            prompt_template=prompt_template,
                            nlg_pipeline = nlg_pipeline
                        )
        df_corpus.at[i,silver_label_technique] = silver_label
        
    
    # Save df_corpus populated with silver_labels
    logger.debug(f"Silver label dist:\n{df_corpus[silver_label_technique].value_counts()}")
    if args.concat_with_train:
        df_train[const.LABEL_DESC_COL] = 'gold_label'
        df_corpus[const.LABEL_DESC_COL] = 'silver_label'
        df_corpus = df_corpus.rename(columns={silver_label_technique: const.LABEL_COL})
        df_corpus = df_corpus[df_corpus[const.LABEL_COL].notna()]
        df_corpus = pd.concat([df_corpus, df_train])
    df_corpus.to_csv(outfile, index=False)