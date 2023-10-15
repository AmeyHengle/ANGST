"""
This module is used for cleaning dataset files and transforming the
input to extract a particular attribute (e.g., the hypothesis-premise
overlap in SNLI).

Each dataset has a parent class in which the cleaning is done and several
subclasses, one for each transformation.
"""
import string
import spacy
import os
import re
import pandas as pd
import random
import logging
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

'''
nohup python3 -u augment.py \
--dataset civilcomments \
--raw_data_dir ../dataset/v_info > ../logs/v_info/augment_civilcomments_val_test.log &
'''

# ------------------------------------------------------- CivilComments -------------------------------------------------------

class CivilCommentsTransformation(object):
    def __init__(self, name, output_dir):
        self.data = pd.read_csv("/projects/ogma3/atharvak/11-830-project-internal/dataset/civilcomments-subsamples/civilcomments_coarse_stratified_subsample_005.csv")
        self.metadata = pd.read_csv("/projects/ogma3/atharvak/11-830-project-internal/dataset/civilcomments-subsamples/metadata_civilcomments_coarse_stratified_subsample_005.csv")
        self.data = pd.concat([self.data, self.metadata], axis=1)[['comment_text', 'split', 'y', 'a']].rename({"comment_text" : "sentence1", "y" : "label"}, axis=1)
        self.data = self.data[self.data['split'] != 0].reset_index(drop=True)
        self.name = name
        self.output_dir = output_dir

    def transformation(self, example):
        raise NotImplementedError

    def transform(self):
        logging.info(f'Applying {self.name} to CivilComments')
        self.data.apply(self.transformation, axis=1).to_csv(
            os.path.join(self.output_dir, f'civilcomments_{self.name}.csv'), index=False)
        

class CivilCommentsStandardTransformation(CivilCommentsTransformation):
    def __init__(self, output_dir):
        super().__init__('std', output_dir)
        print("\nStd transformation...\n")

    def transformation(self, example):
        return example
         

class CivilCommentsNullTransformation(CivilCommentsTransformation):
    def __init__(self, output_dir):
        super().__init__('null', output_dir)
        print("\nNull transformation...\n")

    def transformation(self, example):
        example['sentence1'] = " " # using only empty string can yield problems
        return example


class CivilCommentsVocabTransformation(CivilCommentsTransformation):
    def __init__(self, output_dir):
        super().__init__('bad_vocab', output_dir)
        # potentially offensive words were manually selected
        self.bad_words = [ 'nigga', 'niggas', 'niggah', 'niggahs', 'hoe', 'hoes', 'bitch', 'bitches', 'whitey', 'white trash', 'cracker', 'crackers', 'beaner', 'beaners',
            'pussy', 'pussies', 'fag', 'fags', 'faggot', 'faggots', 'ho', 'hos', 'redneck', 'rednecks', 'porn', 'fuck', 'fucks', 'fucker', 'fuckers', 'motherfucker', 
            'motherfuckers', 'nigger', 'niggers', 'coon', 'coons', 'niggaz', 'nig', 'nigs', 'slut', 'sluts', 'wigger', 'wiggers', 'fucked', 'fucking', 'wigga', 'wiggas',
            'retard', 'retards', 'retarded' ] 
        print("\nBad vocab transformation...\n")

    def transformation(self, example):
        pattern = re.compile(rf"(?!\b({'|'.join(self.bad_words)})\b)\b[^ ]+\b", re.IGNORECASE)
        example['sentence1'] = re.sub(pattern, "", example['sentence1'])
        example['sentence1'] = example['sentence1'].translate(str.maketrans('', '', string.punctuation))
        example['sentence1'] = example['sentence1'].strip()

        if example['sentence1'] == "":
            example['sentence1'] = ' ' #using only empty string can yield problems

        return example


class CivilCommentsSentimentVocabTransformation(CivilCommentsTransformation):
    def __init__(self, output_dir):
        super().__init__('sentiment_vocab', output_dir)
        self.bad_vocab = CivilCommentsVocabTransformation(output_dir)
        print("\nSentiment Vocab transformation...\n")

    def transformation(self, example):
        polarity = nlp(example['sentence1'])._.polarity 

        if -0.10 <= polarity <= 0.10:
            sentiment = 'neutral'
        elif polarity > 0.10:
            sentiment = 'positive'
        else:
            sentiment = 'negative'

        example['sentence1'] = ' '.join([sentiment, self.bad_vocab.transformation(example)['sentence1']])

        if example['sentence1'] == "":
            example['sentence1'] = ' ' #using only empty string can yield problems

        return example


class CivilCommentsSentimentTransformation(CivilCommentsTransformation):
    def __init__(self, output_dir):
        super().__init__('sentiment', output_dir)
        self.bad_vocab = CivilCommentsVocabTransformation(output_dir)
        print("\nSentiment transformation...\n")

    def transformation(self, example):
        polarity = nlp(example['sentence1'])._.polarity 

        if -0.10 <= polarity <= 0.10:
            sentiment = 'neutral'
        elif polarity > 0.10:
            sentiment = 'positive'
        else:
            sentiment = 'negative'

        example['sentence1'] = sentiment

        return example
    
    
# ------------------------------------------------------- Davidson -------------------------------------------------------
        
class DWMWTransformation(object):
    def __init__(self, name, output_dir):
        self.data = pd.read_csv('data/dwmw/labeled_data.csv').rename({"tweet" : "sentence1", "class" : "label"}, axis=1)
        self.name = name
        self.output_dir = output_dir

    def transformation(self, example):
        raise NotImplementedError

    def transform(self):
        logging.info(f'Applying {self.name} to DWMW')
        self.data.apply(self.transformation, axis=1).to_csv(
            os.path.join(self.output_dir, f'dwmw_{self.name}.csv'), index=False)


class DWMWStandardTransformation(DWMWTransformation):
    def __init__(self, output_dir):
        super().__init__('std', output_dir)

    def transformation(self, example):
        return example
         

class DWMWNullTransformation(DWMWTransformation):
    def __init__(self, output_dir):
        super().__init__('null', output_dir)

    def transformation(self, example):
        example['sentence1'] = " " # using only empty string can yield problems
        return example


class DWMWVocabTransformation(DWMWTransformation):
    def __init__(self, output_dir):
        super().__init__('bad_vocab', output_dir)
        # potentially offensive words were manually selected
        self.bad_words = [ 'nigga', 'niggas', 'niggah', 'niggahs', 'hoe', 'hoes', 'bitch', 'bitches', 'whitey', 'white trash', 'cracker', 'crackers', 'beaner', 'beaners',
            'pussy', 'pussies', 'fag', 'fags', 'faggot', 'faggots', 'ho', 'hos', 'redneck', 'rednecks', 'porn', 'fuck', 'fucks', 'fucker', 'fuckers', 'motherfucker', 
            'motherfuckers', 'nigger', 'niggers', 'coon', 'coons', 'niggaz', 'nig', 'nigs', 'slut', 'sluts', 'wigger', 'wiggers', 'fucked', 'fucking', 'wigga', 'wiggas',
            'retard', 'retards', 'retarded' ] 

    def transformation(self, example):
        pattern = re.compile(rf"(?!\b({'|'.join(self.bad_words)})\b)\b[^ ]+\b", re.IGNORECASE)
        example['sentence1'] = re.sub(pattern, "", example['sentence1'])
        example['sentence1'] = example['sentence1'].translate(str.maketrans('', '', string.punctuation))
        example['sentence1'] = example['sentence1'].strip()

        if example['sentence1'] == "":
            example['sentence1'] = ' ' #using only empty string can yield problems

        return example


class DWMWSentimentVocabTransformation(DWMWTransformation):
    def __init__(self, output_dir):
        super().__init__('sentiment_vocab', output_dir)
        self.bad_vocab = DWMWVocabTransformation(output_dir)

    def transformation(self, example):
        polarity = nlp(example['sentence1'])._.polarity 

        if -0.10 <= polarity <= 0.10:
            sentiment = 'neutral'
        elif polarity > 0.10:
            sentiment = 'positive'
        else:
            sentiment = 'negative'

        example['sentence1'] = ' '.join([sentiment, self.bad_vocab.transformation(example)['sentence1']])

        if example['sentence1'] == "":
            example['sentence1'] = ' ' #using only empty string can yield problems

        return example


class DWMWSentimentTransformation(DWMWTransformation):
    def __init__(self, output_dir):
        super().__init__('sentiment', output_dir)
        self.bad_vocab = DWMWVocabTransformation(output_dir)

    def transformation(self, example):
        polarity = nlp(example['sentence1'])._.polarity 

        if -0.10 <= polarity <= 0.10:
            sentiment = 'neutral'
        elif polarity > 0.10:
            sentiment = 'positive'
        else:
            sentiment = 'negative'

        example['sentence1'] = sentiment

        return example


if __name__ == "__main__":
    
    parser.add_argument('--dataset', help='dataset to be used.', required=True, type=str, default='civilcomments', choices=['civilcomments', 'davidson'])
    parser.add_argument('--raw_data_dir', help='raw_data directory', required=True, type=str)
    args = parser.parse_args()
    data_dir = os.path.join(args.raw_data_dir, args.dataset)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if args.dataset == 'civilcomments':
        CivilCommentsStandardTransformation(data_dir).transform()
        CivilCommentsNullTransformation(data_dir).transform()
        CivilCommentsVocabTransformation(data_dir).transform()
        CivilCommentsSentimentVocabTransformation(data_dir).transform()
        CivilCommentsSentimentTransformation(data_dir).transform()
    
    elif args.dataset == 'civilcomments':
        DWMWStandardTransformation(data_dir).transform()
        DWMWNullTransformation(data_dir).transform()
        DWMWVocabTransformation(data_dir).transform()
        DWMWSentimentVocabTransformation(data_dir).transform()
        DWMWSentimentTransformation(data_dir).transform()