"""
python bm25.py \
    --query=data/corpus/unlabeled_corpus.csv \
    --corpus=data/train/v0.0.0.csv \
    --outfile=data/mappings/bm25-example.json \
    --topk=5 \
    ;
"""


import os
import json
import argparse
import pandas as pd
import numpy as np
import constants as const
from tqdm import tqdm
from loguru import logger
from rank_bm25 import BM25Okapi
from utils import clean_text, NpEncoder


def get_examplars_bm25(
    datapath_corpus, 
    datapath_query,
    text_col,
    id_col,
    outfile,
    topk
):
    logger.debug("\nFetching query and corpus files")
    df_corpus = pd.read_csv(datapath_corpus, usecols=[text_col, id_col])
    df_corpus = df_corpus[df_corpus[text_col].notna()]
    df_corpus = df_corpus[df_corpus[id_col].notna()]
    df_query = pd.read_csv(datapath_query, usecols=[text_col, id_col])
    df_query = df_query[df_query[text_col].notna()]
    df_query = df_query[df_query[id_col].notna()]
    logger.debug(f"\n\nQuery size: {df_query.shape}\nCorpus size: {df_corpus.shape}")


    logger.debug(f"\nNormalizing {text_col} col")
    df_corpus[text_col] = df_corpus[text_col].apply(lambda x: clean_text(x))
    df_query[text_col] = df_query[text_col].apply(lambda x: clean_text(x))


    corpus = df_corpus[text_col].values.tolist()
    query_ids = df_query[id_col].values.tolist()
    exemplar_dict = dict.fromkeys(query_ids,[])


    logger.debug(f"\nTokenizing corpus")
    corpus_tokenized = [doc.split(" ") for doc in corpus]

    logger.debug(f"\nTokenizing queries")
    query_tokenized = [doc.split(" ") for doc in df_query[text_col].values.tolist()]


    logger.debug(f"\nGetting Exemplars")
    bm25 = BM25Okapi(corpus_tokenized)
    for i in tqdm(range(len(query_tokenized)),desc="Running BM25"):
        query = query_tokenized[i]
        top_n = bm25.get_top_n(query, corpus, n=topk)
        scores = np.sort(bm25.get_scores(query))[::-1][:topk]
        
        for match,score in zip(top_n,scores):
            query_id = query_ids[i]
            corpus_id = df_corpus[df_corpus[text_col] == match].iloc[0][id_col]
            if exemplar_dict[query_id]:
                exemplar_dict[query_id].append((corpus_id,score))
            else:
                exemplar_dict[query_id] = []
                exemplar_dict[query_id].append((corpus_id,score))
            

    logger.debug(f"\Saving Exemplars to {outfile}")
    if os.path.exists(outfile):
        logger.debug(f"{outfile} already exists, deleting")
        os.remove(outfile)
    
    with open(outfile, "w") as write_file:
        json.dump(exemplar_dict, write_file, indent=4, ensure_ascii=False, cls=NpEncoder)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--query")
    parser.add_argument("--corpus")
    parser.add_argument("--outfile")
    parser.add_argument("--text_col", default=const.TEXT_COL)
    parser.add_argument("--id_col", default=const.ID_COL)
    parser.add_argument("--topk", default=const.CONFIG_BM25['topk'])
    args = parser.parse_args()
    
    datapath_corpus = args.corpus
    datapath_query = args.query
    text_col = args.text_col
    id_col = args.id_col
    outfile = args.outfile
    topk = int(args.topk)

    if not outfile.endswith('.json'):
        raise Exception(
            "\noutfile must have a {.json} extension. View --help"
        )
    
    get_examplars_bm25(
        datapath_corpus, 
        datapath_query,
        text_col,
        id_col,
        outfile,
        topk
    )