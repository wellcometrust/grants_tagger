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

import pandas as pd

# TODO: Fix keys for calculating human accuracy

def yield_preprocess_data(
        data,
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
    processed_data = data.drop_duplicates(subset=["Grant_ID", "Sciencetags"])
    processed_data = processed_data.dropna(subset=["Synopsis"])
    processed_data = processed_data.groupby('Grant_ID').agg({
        'Sciencetags': lambda x: ",".join(x),
        'Title': lambda x: x.iloc[0],
        'Synopsis': lambda x: x.iloc[0],
        'Lay Summary': lambda x: x.iloc[0],
        'Qu.': lambda x: x.iloc[0],
        'Scheme': lambda x: x.iloc[0],
        'Team': lambda x: x.iloc[0]
    }).reset_index()
    processed_data = processed_data[processed_data["Synopsis"] != "No Data Entered"]
    processed_data = processed_data[processed_data["Sciencetags"] != "No tag"]
    processed_data = processed_data[processed_data["Sciencetags"] != "Not enough information available"]
    processed_data = processed_data.drop_duplicates(subset="Grant_ID")
    processed_data = processed_data.drop_duplicates(subset="Synopsis")
    processed_data = processed_data.replace("No Data Entered", "").fillna("")

    for processed_item in processed_data.to_dict("records"):
        yield {
            'text': " ".join([processed_item[col] for col in text_cols]),
            'tags': processed_item["Sciencetags"].split(','),
            'meta': {col: processed_item[col] for col in meta_cols}
        }

def preprocess(input_path, output_path, text_cols, meta_cols):
    data = pd.read_excel(input_path)

    tags = []
    with open(output_path, 'w') as f:
        for chunk in yield_preprocess_data(data, text_cols, meta_cols):
            f.write(json.dumps(chunk))
            f.write('\n')
            tags.append(chunk['tags'])

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--input', type=Path, help="path to raw Excel file with tagged or untagged grant data")
    argparser.add_argument('--output', type=Path, help="path to JSONL output file that will be generated")
    argparser.add_argument('--text_cols', type=str, default="Title,Synopsis", help="comma delimited column names to concatenate to text")
    argparser.add_argument('--meta_cols', type=str, default="Grant_ID,Title", help="comma delimited column names to include in the meta")
    argparser.add_argument('--config', type=Path, help="path to config file that defines the arguments")
    args = argparser.parse_args()

    if args.config:
        cfg = ConfigParser(allow_no_value=True)
        cfg.read(args.config)

        input_path = cfg["preprocess"]["input"]
        output_path = cfg["preprocess"]["output"]
        text_cols = cfg["preprocess"]["text_cols"]
        meta_cols = cfg["preprocess"].get("meta_cols", "Grant_ID,Title")
    else:
        input_path = args.input
        output_path = args.output
        text_cols = args.text_cols
        meta_cols = args.meta_cols

    text_cols = text_cols.split(",")
    meta_cols = meta_cols.split(",")
    if os.path.exists(output_path):
        print(f"{output_path} exists. Remove if you want to rerun.")
    else:
        preprocess(input_path, output_path, text_cols, meta_cols)
