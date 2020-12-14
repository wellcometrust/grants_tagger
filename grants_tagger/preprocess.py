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
    processed_data["intersection"] = processed_data["intersection"].str.replace("19: Maternal, Child", "19: Maternal Child")
    processed_data["intersection"] = processed_data["intersection"].str.replace("24: Data Science, Computational", "24: Data Science Computational")
    processed_data["Tagger 1 only"] = processed_data["Tagger 1 only"].str.replace("19: Maternal, Child", "19: Maternal Child")
    processed_data["Tagger 1 only"] = processed_data["Tagger 1 only"].str.replace("24: Data Science, Computational", "24: Data Science Computational")
    processed_data["Tagger 1 only"] = processed_data["Tagger 1 only"].str.replace("12: Brain Cells, Circuits", "12: Brain Cells Circuits")
    processed_data["Tagger1_tags"] = processed_data.apply(
        lambda x: x['intersection'].strip("[]").replace("'","").split(', ') + x['Tagger 1 only'].strip('[]').split(', '),
        axis=1
    )
    processed_data = processed_data.groupby('Grant_ID').agg({
        'Sciencetags': lambda x: ",".join(x),
        'Title': lambda x: x.iloc[0],
        'Synopsis': lambda x: x.iloc[0],
        'Lay Summary': lambda x: x.iloc[0],
        'Qu.': lambda x: x.iloc[0],
        'Scheme': lambda x: x.iloc[0],
        'Team': lambda x: x.iloc[0],
        'Tagger1_tags': lambda x: x.iloc[0]
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
