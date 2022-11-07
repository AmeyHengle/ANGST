import pandas as pd
import numpy as np
import glob
import functools as ft
from loguru import logger
import os
import json
import constants as const
from utils import clean_text, split_dataset, set_random_seed

SEED = const.RANDOM_STATE
set_random_seed(SEED)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = const.REDDIT_COLS + const.DISORDER_COLS
    df = df.rename(columns={"Sr. No.": "srno", "serial_no": "srno", "Title": "title", "post": "selftext", "Depressive Disorders": "label_depressive_disorder", "Anxiety Disorders": "label_anxiety_disorder"})    
    df = df[keep_cols]
    df = df.fillna(0)
    
    def process_label_col(label):
        try:
            label = float(label)
            return label
        except:
            return float(0)

    for lc in const.DISORDER_COLS:
        df[lc] = df[lc].apply(lambda x: process_label_col(x))
        logger.debug(f"{df[lc].value_counts()}")
        
    return df


def transform(label_array):
    output = set()
    for l in label_array:
        if label_array[0] == 1:
            output.add("depressive_disorder")
        if label_array[1] == 1:
            output.add("anxiety_disorder")
    
    return output if len(output) > 0 else set(["control_group"])


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df['dist_depressive_disorder'] = None
    df['dist_anxiety_disorder'] = None

    for i in range(df.shape[0]):
        df.at[i, 'dist_depressive_disorder'] = \
        [
            df.iloc[i]['label_depressive_disorder_a1'],
            df.iloc[i]['label_depressive_disorder_a2'],
            df.iloc[i]['label_depressive_disorder_a3']
        ]
        
        df.at[i, 'dist_anxiety_disorder'] = \
        [
            df.iloc[i]['label_anxiety_disorder_a1'],
            df.iloc[i]['label_anxiety_disorder_a2'],
            df.iloc[i]['label_anxiety_disorder_a3']
        ]

    df['mv_depressive_disorder'] = df['dist_depressive_disorder'].apply(lambda x: int(max(set(x), key=x.count)))
    df['mv_anxiety_disorder'] = df['dist_anxiety_disorder'].apply(lambda x: int(max(set(x), key=x.count)))
    df['a1t_depressive_disorder'] = df['dist_depressive_disorder'].apply(lambda x: 1 if 1 in x else 0)
    df['a1t_anxiety_disorder'] = df['dist_anxiety_disorder'].apply(lambda x: 1 if 1 in x else 0)

    df['labels_mv'] = [[x,y] for x,y in zip(df['mv_depressive_disorder'], df['mv_anxiety_disorder'])]
    df['labelsdict_mv'] = df['labels_mv'].apply(lambda x: transform(x))
    df['labels_a1t'] = [[x,y] for x,y in zip(df['a1t_depressive_disorder'], df['a1t_anxiety_disorder'])]
    df['labelsdict_a1t'] = df['labels_a1t'].apply(lambda x: transform(x))
    
    return df


if __name__ == "__main__":
    
    dfs = []
    df_merged = pd.DataFrame()
    df_aggregated = pd.DataFrame()
    
    """
    Step1:
    Process raw annotation files. 
    """
    for fname in glob.iglob(f'{const.ANNOTATIONS_RAW}/*.csv'):
        logger.debug(f"Processing {fname}")
        df = pd.read_csv(fname)
        df = preprocess(df)
        dfs.append(df)

    """
    Step2:
    Merge preprocessed files of each annotator. 
    Derive final label column based on majority voting (atleast 2)
    Derive final label column based on union (atleast 1)
    Store final annotation files to /home/amey/projects/mentalBert/data/annotations/processed
    """
    for df in dfs:
        df = df.drop_duplicates(subset=[const.ID_COL])
        if df_merged.shape[0] > 0:
            df_merged = pd.merge(
                df_merged, df, on=[const.ID_COL]    
            )
        else:
            df_merged = df
        
    df_merged = df_merged.rename(
        columns = {
            "label_depressive_disorder": "label_depressive_disorder_a1",
            "label_anxiety_disorder": "label_anxiety_disorder_a1",
            "label_depressive_disorder_x": "label_depressive_disorder_a2",
            "label_anxiety_disorder_x": "label_anxiety_disorder_a2",
            "label_depressive_disorder_y": "label_depressive_disorder_a3",
            "label_anxiety_disorder_y": "label_anxiety_disorder_a3"
        }
    )
    logger.debug(f"Merged df: {df_merged.shape}")

    df_aggregated = aggregate(df_merged)
    logger.debug(f"Aggregate df: {df_aggregated.shape}")

    df_aggregated.to_csv(os.path.join(const.ANNOTATIONS_PROC, 'annotations_agg.csv'),index=False)

    use_cols = ["id", "title","selftext", "labels", "disorder", const.TEXT_COL]
    df_final  = df_aggregated.rename(columns={
        "labelsdict_mv" : "disorder",
        "labels_mv" : "labels"
    })
    df_final[const.TEXT_COL] = df_final['title'] + '.\n' + df_final['selftext']
    df_final[const.TEXT_COL] = df_final[const.TEXT_COL].apply(lambda x: clean_text(x))

    df_final[use_cols].to_csv(os.path.join(const.ANNOTATIONS_PROC, 'annotations_final.csv'),index=False)
    logger.debug(f"Saved to {os.path.join(const.ANNOTATIONS_PROC)}")
    
    # Get train - test split
    train, test = split_dataset(
                    df_final, 
                    test_size=const.TEST_SIZE, 
                    id_col=const.ID_COL, 
                    stratify_col=const.LABEL_COL,
                    random_state=const.RANDOM_STATE
                )
    
    train[use_cols].to_csv(os.path.join(const.TRAIN, 'v0.0.0.csv'),index=False)
    logger.debug(f"Saved to {os.path.join(const.TRAIN)}")

    test[use_cols].to_csv(os.path.join(const.TEST, 'test.csv'),index=False)
    logger.debug(f"Saved to {os.path.join(const.TEST)}")
