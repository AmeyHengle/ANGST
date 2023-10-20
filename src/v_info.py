from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tokenizers.pre_tokenizers import Whitespace
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()


def v_entropy(data_fn, model, tokenizer, input_key='sentence1', batch_size=100):
    """
    Calculate the V-entropy (in bits) on the data given in data_fn. This can be
    used to calculate both the V-entropy and conditional V-entropy (for the
    former, the input column would only have null data and the model would be
    trained on this null data).

    Args:
        data_fn: path to data; should contain the label in the 'label' column
        model: path to saved model or model name in HuggingFace library
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        input_key: column name of X variable in data_fn
        batch_size: data batch_size

    Returns:
        Tuple of (V-entropies, correctness of predictions, predicted labels).
        Each is a List of n entries (n = number of examples in data_fn).
    """
    # added for gpt2 
    if tokenizer == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)

    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True, device=0, padding=True, truncation=True)
    data = pd.read_csv(data_fn)
    
    entropies = []
    correct = []
    predicted_labels = []

    for j in tqdm(range(0, len(data), batch_size)):
        batch = data[j:j+batch_size]
        predictions = classifier(batch[input_key].tolist())

        for i in range(len(batch)):
            # prob = next(d for d in predictions[i] if d[args.label] == batch.iloc[i][args.label])['score']
            prob = next(d for d in predictions[i] if d['label'] == batch.iloc[i][args.label])['score']
            entropies.append(-1 * np.log2(prob))
            predicted_label = max(predictions[i], key=lambda x: x['score'])['label'] 
            predicted_labels.append(predicted_label)
            correct.append(predicted_label == batch.iloc[i][args.label])

    torch.cuda.empty_cache()

    return entropies, correct, predicted_labels


def v_info(data_fn, model, null_data_fn, null_model, tokenizer, out_fn="", input_key='sentence1'):
    """
    Calculate the V-entropy, conditional V-entropy, and V-information on the
    data in data_fn. Add these columns to the data in data_fn and return as a 
    pandas DataFrame. This means that each row will contain the (pointwise
    V-entropy, pointwise conditional V-entropy, and pointwise V-info (PVI)). By
    taking the average over all the rows, you can get the V-entropy, conditional
    V-entropy, and V-info respectively.

    Args:
        data_fn: path to data; should contain the label in the 'label' column 
            and X in column specified by input_key
        model: path to saved model or model name in HuggingFace library
        null_data: path to null data (column specified by input_key should have
            null data)
        null_model: path to saved model trained on null data
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        out_fn: where to saved 
        input_key: column name of X variable in data_fn 

    Returns:
        Pandas DataFrame of the data in data_fn, with the three additional 
        columns specified above.
    """
    data = pd.read_csv(data_fn)
    data['H_yb'], _, _ = v_entropy(null_data_fn, null_model, tokenizer, input_key=input_key) 
    data['H_yx'], data['correct_yx'], data['predicted_label'] = v_entropy(data_fn, model, tokenizer, input_key=input_key)
    data['PVI'] = data['H_yb'] - data['H_yx']

    if out_fn:
        data.to_csv(out_fn, index=False)

    return data


def find_annotation_artefacts(data_fn, model, tokenizer, input_key='sentence1', min_freq=5, pre_tokenize=True):
    """
    Find token-level annotation artefacts (i.e., tokens that once removed, lead to the
    greatest decrease in PVI for each class).

    Args:
        data_fn: path to data; should contain the label in the 'label' column 
            and X in column specified by input_key
        model: path to saved model or model name in HuggingFace library
        tokenizer: path to tokenizer or tokenizer name in HuggingFace library
        input_key: column name of X variable in data_fn 
        min_freq: minimum number of times a token needs to appear (in a given class' examples)
            to be considered a potential partefact
        pre_tokenize: if True, do not consider subword-level tokens (each word is a token)

    Returns:
        A pandas DataFrame with one column for each unique label and one row for each token.
        The value of the entry is the entropy delta (i.e., the drop in PVI for that class if that
        token is removed from the input). If the token does meet the min_freq threshold, then the
        entry is empty.
    """
    data = pd.read_csv(data_fn)
    labels = [ l for l in data[args.label].unique().tolist() if l >= 0 ] # assume labels are numbers
    token_entropy_deltas = { l : {} for l in labels }
    all_tokens = set([])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    pre_tokenizer = Whitespace()

    # added for gpt2 
    if tokenizer == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model, pad_token_id=tokenizer.eos_token_id)

    # get the PVI for each example
    print("Getting conditional V-entropies ...")
    entropies, _ = v_entropy(data_fn, model, tokenizer, input_key=input_key)

    print("Calculating token-wise delta for conditional entropies ...")
    classifier = pipeline('text-classification', model=model, tokenizer=tokenizer, return_all_scores=True, device=0)

    for i in tqdm(range(len(data))):
        example = data.iloc[i]

	    # mislabelled examples; ignore these
        if example[args.label] < 0:
            continue

        if pre_tokenize:
            tokens = [ t[0] for t in pre_tokenizer.pre_tokenize_str(example['sentence1']) ]
        else:
            tokens = tokenizer.tokenize(example['sentence1'])

        # create m versions of the input in which one of the m tokens it contains is omitted
        batch = pd.concat([ example ] * len(tokens), axis=1).transpose()
         
        for j in range(len(tokens)):
            # create new input by omitting token j
            batch.iloc[j][input_key] = tokenizer.convert_tokens_to_string(tokens[:j] + tokens[j+1:])
            all_tokens.add(tokens[j])

            for label in labels:
                if tokens[j] not in token_entropy_deltas[label]: token_entropy_deltas[label][tokens[j]] = []

        # get the predictions (in rare cases, have to split the batch up into mini-batches of 100 because it's too large)
        predictions = []
        for k in range(0, len(batch), 100):
            predictions.extend(classifier(batch[input_key][k:k+100].tolist()))
        
        for j in range(len(tokens)):
            # prob = next(d for d in predictions[j] if d[args.label] == example[args.label])['score']
            prob = next(d for d in predictions[j] if d['label'] == example[args.label])['score']
            entropy_delta = (-1 * np.log2(prob)) - entropies[i]
            token_entropy_deltas[example[args.label]][tokens[j]].append(entropy_delta)

    torch.cuda.empty_cache()

    total_freq = { t : sum(len(token_entropy_deltas[l][t]) for l in labels) for t in all_tokens }
    # average over all instances of token in class
    for label in labels:
        for token in token_entropy_deltas[label]:
            if total_freq[token] > min_freq:
            	token_entropy_deltas[label][token] = np.nanmean(token_entropy_deltas[label][token]) 
            else:
                token_entropy_deltas[label][token] = np.nan

    table = pd.DataFrame.from_dict(token_entropy_deltas)
    return table


if __name__ == "__main__":
    parser.add_argument('--dataset', help='dataset to be used.', required=True, type=str, default='MHCD', choices=['MHCD', 'DATD', 'dreddit', 'dep_reddit', 'SDCNL'])
    parser.add_argument("--model", type=str, default='roberta-base', help="the model to use")
    parser.add_argument("--label", type=str, default='label', help="The label to choose")
    parser.add_argument('--dataset_dir', help='dataset directory', required=True, type=str)
    parser.add_argument('--model_dir', help='model directory', required=True, type=str)
    args = parser.parse_args()

    for key, val in vars(args).items():
        print(f"{key}: {val}")
        
    label = '' if args.label == 'label' else f'-{args.label}'
    print(label)
    print("\n\nCalculating vinfo...\n\n")
    v_info(
        f"{args.dataset_dir}/{args.dataset}_std.csv",
        f"{args.model_dir}/{args.model}-{args.dataset}-std{label}",
        f"{args.dataset_dir}/{args.dataset}_null.csv",
        f"{args.model_dir}/{args.model}-{args.dataset}-null{label}",
        args.model,
        out_fn=f"{args.dataset_dir}/{args.dataset}_{args.model}_std_null_vinfo{label}.csv",
    )
    print("\n\nDone!\n\n")