import os

TRAIN = os.path.join('./','data/train')
TEST = os.path.join('./','data/test')
ROOT = os.path.join('./')
ANNOTATIONS_RAW = os.path.join('./','data/annotations/raw')
ANNOTATIONS_PROC = os.path.join('./','data/annotations/processed')
OPENAI_CREDS = os.path.join('./','creds/openai.json')

REDDIT_COLS = ["id", "title","selftext"]
DISORDER_COLS = ["label_depressive_disorder", "label_anxiety_disorder"]
TEXT_COL = "text"
LABEL_COL = "labels"
ID_COL = "id"
LABEL_DESC_COL = "label_desc"

TEST_SIZE = 0.3
RANDOM_STATE = 1998
MODEL_NAME = "bert-base-uncased"
MODEL_TYPE = "bert"
 

CONFIG_GPT = {
    "model": "text-davinci-002",
    "temperature": 0,
    "max_tokens": 6,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["\n"]
}

CONFIG_SEMANTIC_SIMILARITY = {
    "model_name": 'all-mpnet-base-v2',
    "max_seq_len": 512,
    "topk": 5,
}

CONFIG_BM25 = {
    "topk": 5
}

CONFIG_TRAIN =  {
    'num_train_epochs': 5,
    'max_seq_length': 512,
    'overwrite_output_dir': True,
    'train_batch_size': 32,
    'eval_batch_size': 32,
    'evaluate_during_training' : True,
    'use_multiprocessing': False,
    'use_multiprocessing_for_evaluation': False,
    'save_eval_checkpoints': False,
    'save_model_every_epoch': False,
    'save_optimizer_and_scheduler': False,
    'save_steps': -1,
    'use_early_stopping' : True,
    'evaluate_during_training_verbose' : True,
    'learning_rate' : 2e-5,
    'no_cache': True
}


CONFIG_SILVER_LABELLING = {
    "technique": "ss",
    "use_gpt": False,
    "threshold_ss": 0.7,
    "threshold_dm25": 0.7,
    "topk": 5
}