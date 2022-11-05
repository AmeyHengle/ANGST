"""
python semantic-similarity.py \
    --query=data/corpus/unlabeled_corpus.csv \
    --corpus=data/train/v0.0.0.csv \
    --outfile=data/mappings/semantic-similarity-example.json \
    --model_name=all-mpnet-base-v2 \
    --topk=5 \
    ;
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
import warnings
import torch
import random
from sentence_transformers import SentenceTransformer, util
warnings.filterwarnings("ignore")

from utils import clean_text, NpEncoder
import constants as const


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")
    

def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999
    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_examplars_st(
    datapath_corpus, 
    datapath_query,
    text_col,
    id_col,
    outfile,
    model_name,
    max_seq_len,
    topk,
):
    
    logger.debug("\nFetching query and corpus files")
    df_corpus = pd.read_csv(datapath_corpus, usecols=[text_col, id_col])
    df_corpus = df_corpus[df_corpus[text_col].notna()]
    df_corpus = df_corpus[df_corpus[id_col].notna()]
    df_query = pd.read_csv(datapath_query, usecols=[text_col, id_col])
    df_query = df_query[df_query[text_col].notna()]
    df_query = df_query[df_query[id_col].notna()]
    logger.debug(f"\nQuery size: {df_query.shape}\nCorpus size: {df_corpus.shape}")


    logger.debug(f"\nNormalizing {text_col} col")
    df_corpus[text_col] = df_corpus[text_col].apply(lambda x: clean_text(x))
    df_query[text_col] = df_query[text_col].apply(lambda x: clean_text(x))
    
    corpus = df_corpus[text_col].values.tolist()
    queries = df_query[text_col].values.tolist()
    query_ids = df_query[id_col].values.tolist()
    exemplar_dict = dict.fromkeys(query_ids,[])


    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_len
    model.to(DEVICE)


    logger.debug(f"\nCalculating corpus embeddings")
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    corpus_embeddings = corpus_embeddings.to(DEVICE)

    logger.debug(f"\nCalculating query embeddings")
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    query_embeddings = query_embeddings.to(DEVICE)

    logger.debug(f"\nGetting top {topk} matches for each query from corpus")
    matches = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.cos_sim, top_k=topk)


    logger.debug(f"\nGenerating exemplar_dict")
    if len(matches) != len(queries):
        raise Exception(
            f"Size mismatch: Matches {len(matches)} != Queries {len(queries)}"
        )
             
    for i in tqdm(range(len(matches))):
        query_id = df_query.iloc[i][id_col]
        exemplar_dict[query_id] = [(df_corpus.iloc[x['corpus_id']][id_col],x['score']) for x in matches[i]]
            
    if len(exemplar_dict) != df_query.shape[0]:
        raise Exception(
            f"Size mismatch: Query {df_query.shape[0]} != Exemplar dict {len(exemplar_dict)}"
        )


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
    parser.add_argument("--model_name", default=const.CONFIG_SEMANTIC_SIMILARITY['model_name'])
    parser.add_argument("--max_seq_len", default=const.CONFIG_SEMANTIC_SIMILARITY['max_seq_len'])
    parser.add_argument("--topk", default=const.CONFIG_SEMANTIC_SIMILARITY['topk'])
    
    args = parser.parse_args()
    
    datapath_corpus = args.corpus
    datapath_query = args.query
    text_col = args.text_col
    id_col = args.id_col
    outfile = args.outfile
    model_name = args.model_name
    max_seq_len = int(args.max_seq_len)
    topk = int(args.topk)
    
    if not outfile.endswith('.json'):
        raise Exception(
            "\nOutfile must have a {.json} extension. View --help"
        )
    
    SEED = const.RANDOM_STATE
    set_random_seed(SEED)
    
    get_examplars_st(
        datapath_corpus, 
        datapath_query,
        text_col,
        id_col,
        outfile,
        model_name,
        max_seq_len,
        topk
    )