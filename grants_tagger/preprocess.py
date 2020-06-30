# encoding: utf-8
"""
Preprocesses the raw Excel files containing grant data with or without tags
before passing on to train or teach(by Prodigy)
"""

from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
import pickle
import json
import os

from nltk.stem import WordNetLemmatizer
import pandas as pd


def lemmatize(text):
    lemmer = WordNetLemmatizer()
    return " ".join([lemmer.lemmatize(word, 'v') for word in text.split()])

def filter_missing_data(data):
    if 'Sciencetags' not in data:
        data['Sciencetags'] = ""
    data.rename(columns={'Sciencetags': 'tags'}, inplace=True)
    data = data[data['tags'] != 'No tag']
    data = data[data['tags'] != 'Not enough information available']
    return data

def clean_text(row, lemmatize_func, text_cols, for_prodigy=False):
    if for_prodigy:
        title = row['Title'].upper()
        synopsis = row['Synopsis'].replace('\n','')
        text = f"{title}\n\n{synopsis}"
    else:
        text = " ".join(row[text_cols].tolist())
    return lemmatize_func(text)

def create_grant_text(data, lemmatize_func, text_cols):
    "Return dataframe of Grant ID, text"
    grant_data = data.copy()
    grant_data['text'] = grant_data.apply(lambda x: clean_text(x, lemmatize_func, text_cols), axis=1)
    grant_data = grant_data.drop_duplicates(subset = 'Grant_ID')
    grant_data = grant_data.drop_duplicates(subset = 'Synopsis') # Drops Synopsis with No Data Entered
    return grant_data[['Grant_ID', 'text']]

def create_tagger_metadata(data, tagger_level):
    tagger_tags_col = 'Tagger{}_tags'.format(tagger_level)
    if 'Tagger {} only'.format(tagger_level) not in data.columns:
        return data['Grant_ID'].drop_duplicates()
    # create columns that define which tags each tagger used ('intersection' is the column indicating tags where both taggers agreed)
    data[tagger_tags_col] = data['intersection'] + data['Tagger {} only'.format(tagger_level)]
    data[tagger_tags_col] = data.apply(lambda x: x['tags'] if x['tags'] in x[tagger_tags_col] else '', axis=1)
    d = data.groupby('Grant_ID')[tagger_tags_col].apply(lambda x: ','.join(x)).reset_index()
    return d[['Grant_ID', tagger_tags_col]]

def create_grant_metadata(data, meta_cols):
    "Returns dataFrame containing Grant ID and metadata cols"
    grant_metadata = data[meta_cols].drop_duplicates(subset = 'Grant_ID')
    grant_tagger_1 = create_tagger_metadata(data, 1)
    grant_tagger_2 = create_tagger_metadata(data, 2)
    return grant_metadata.merge(grant_tagger_1, on='Grant_ID').merge(grant_tagger_2, on='Grant_ID')

def create_grant_tags(data):
    "Returns dataframe of Grant ID, comma separated tags"
    grant_tags = data.groupby('Grant_ID')['tags'].apply(lambda x: ','.join(x)).reset_index()
    return grant_tags

def yield_preprocess_data(
        data,
        lemmatize_func=lambda x: x,
        text_cols=["Title", "Synopsis"],
        meta_cols=["Grant_ID", "Title"]):
    """
    Args:
        data: pandas dataframe with columns file_hash, tag,
            pdf_text, ...
        lemmatize_func
        text_cols: cols to concatenate into text
        meta_cols: cols to return in meta
    Returns:
        text: list of text associated to a grant
        tags: list of comma separated tags per grant
        meta: list of tuple containing meta information for a grant such as its id
    """
    # cols = ['Grant Team', 'ERG', 'Lead Applicant', 'Organisation', 'Scheme', 'Title', 'Synopsis', 'Lay Summary', 'Qu.']
    data = data.dropna(subset=['Synopsis'])
    data = filter_missing_data(data)
    data = data.drop_duplicates(subset=['Grant_ID', 'tags'])
    grant_tags = create_grant_tags(data)
    grant_text = create_grant_text(data, lemmatize_func, text_cols)
    grant_metadata = create_grant_metadata(data, meta_cols)
    grant_data = grant_text.merge(grant_tags, on='Grant_ID').merge(grant_metadata, on='Grant_ID', how='left')

    texts = grant_data['text'].tolist()
    tags = grant_data['tags'].tolist()
    meta = grant_data[grant_metadata.columns].to_dict('records')

    for text, tag, met in zip(texts, tags, meta):
        yield {
            'text': text,
            'tags': tag.split(','),
            'meta': met
        }

def preprocess(input_path, output_path, text_cols):
    data = pd.read_excel(input_path)

    tags = []
    with open(output_path, 'w') as f:
        for chunk in yield_preprocess_data(data, text_cols=text_cols):
            f.write(json.dumps(chunk))
            f.write('\n')
            tags.append(chunk['tags'])

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--input', type=Path, help="path to raw Excel file with tagged or untagged grant data")
    argparser.add_argument('--output', type=Path, help="path to JSONL output file that will be generated")
    argparser.add_argument('--text_cols', type=str, default="Title,Synopsis", help="comma delimited column names to concatenate to text")
    argparser.add_argument('--config', type=Path, help="path to config file that defines the arguments")
    args = argparser.parse_args()
    print(args.config)
    if args.config:
        cfg = ConfigParser(allow_no_value=True)
        cfg.read(args.config)

        input_path = cfg["preprocess"]["input"]
        output_path = cfg["preprocess"]["output"]
        text_cols = cfg["preprocess"]["text_cols"]
    else:
        input_path = args.input
        output_path = args.output
        text_cols = args.text_cols or "Title,Synopsis"

    text_cols = args.text_cols.split(",")
    if os.path.exists(output_path):
        print(f"{output_path} exists. Remove if you want to rerun.")
    else:
        preprocess(input_path, output_path, text_cols)
