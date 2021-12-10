# encoding: utf-8
"""
Preprocesses the raw Excel files containing grant data with or without tags
before passing on to train or teach(by Prodigy)
"""

from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
import pickle
import configparser
import json
import os
import typer
import yaml
import shutil

import pandas as pd
from typing import List, Optional
from grants_tagger.utils import verify_if_paths_exist
from grants_tagger.label_binarizer import create_label_binarizer
from grants_tagger.split_data import split_data


def process_tagger_data(tagger_data):
    tagger_data = tagger_data.replace("19: Maternal, Child", "19: Maternal Child")
    tagger_data = tagger_data.replace(
        "24: Data Science, Computational", "24: Data Science Computational"
    )
    tagger_data = tagger_data.replace(
        "12: Brain Cells, Circuits", "12: Brain Cells Circuits"
    )
    tagger_data = tagger_data.strip("[]").replace("'", "")
    return tagger_data


def yield_preprocess_data(
    data, text_cols=["Title", "Synopsis"], meta_cols=["Grant_ID", "Title"]
):
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
    if "Tagger1_tags" in meta_cols:
        processed_data["intersection"] = processed_data["intersection"].apply(
            lambda x: process_tagger_data(x)
        )
        processed_data["Tagger 1 only"] = processed_data["Tagger 1 only"].apply(
            lambda x: process_tagger_data(x)
        )
        processed_data["Tagger1_tags"] = processed_data.apply(
            lambda x: (x["intersection"] + x["Tagger 1 only"]).split(", "), axis=1
        )
    aggregations = {
        "Sciencetags": lambda x: ",".join(x),
        "Title": lambda x: x.iloc[0],
        "Synopsis": lambda x: x.iloc[0],
        "Lay Summary": lambda x: x.iloc[0],
        "Qu.": lambda x: x.iloc[0],
        "Scheme": lambda x: x.iloc[0],
        "Team": lambda x: x.iloc[0],
    }
    if "Tagger1_tags" in meta_cols:
        aggregations["Tagger1_tags"] = lambda x: x.iloc[0]
    processed_data = processed_data.groupby("Grant_ID").agg(aggregations).reset_index()
    processed_data = processed_data[processed_data["Synopsis"] != "No Data Entered"]
    processed_data = processed_data[processed_data["Sciencetags"] != "No tag"]
    processed_data = processed_data[
        processed_data["Sciencetags"] != "Not enough information available"
    ]
    processed_data = processed_data.drop_duplicates(subset="Grant_ID")
    processed_data = processed_data.drop_duplicates(subset="Synopsis")
    processed_data = processed_data.replace("No Data Entered", "").fillna("")

    for processed_item in processed_data.to_dict("records"):
        yield {
            "text": " ".join([processed_item[col] for col in text_cols]),
            "tags": processed_item["Sciencetags"].split(","),
            "meta": {col: processed_item[col] for col in meta_cols},
        }


def preprocess(input_path, output_path, text_cols, meta_cols):
    data = pd.read_excel(input_path)

    tags = []
    with open(output_path, "w") as f:
        for chunk in yield_preprocess_data(data, text_cols, meta_cols):
            f.write(json.dumps(chunk))
            f.write("\n")
            tags.append(chunk["tags"])


preprocess_wellcome_app = typer.Typer()


@preprocess_wellcome_app.command()
def preprocess_wellcome_cli(
    input_path: Optional[Path] = typer.Argument(
        None, help="path to raw Excel file with tagged or untagged grant data"
    ),
    train_output_path: Optional[str] = typer.Argument(
        None, help="path to JSONL output file that will be generated for the train set"
    ),
    label_binarizer_path: Optional[Path] = typer.Argument(
        None, help="path to pickle file that will contain the label binarizer"
    ),
    test_output_path: Optional[str] = typer.Option(
        None, help="path to JSONL output file that will be generated for the test set"
    ),
    text_cols: Optional[str] = typer.Option(
        None, help="comma delimited column names to concatenate to text"
    ),
    meta_cols: Optional[str] = typer.Option(
        None, help="comma delimited column names to include in the meta"
    ),
    test_split: Optional[float] = typer.Option(
        0.1, help="split percentage for test data. if None no split."
    ),
    config: Path = typer.Option(
        None, help="path to config file that defines the arguments"
    ),
):

    params_path = os.path.join(os.path.dirname(__file__), "../params.yaml")
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # Default values from params
    if not text_cols:
        text_cols = params["preprocess_wellcome_science"]["text_cols"]
    if not meta_cols:
        meta_cols = params["preprocess_wellcome_science"]["meta_cols"]

    # Note that config overides values if provided, this ensures backwards compatibility
    if config:
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)

        input_path = cfg["preprocess"]["input"]
        train_output_path = cfg["preprocess"]["output"]
        text_cols = cfg["preprocess"]["text_cols"]
        if not text_cols:
            text_cols = "Title,Synopsis"
        meta_cols = cfg["preprocess"].get("meta_cols", "Grant_ID,Title")

    text_cols = text_cols.split(",")
    meta_cols = meta_cols.split(",")

    if verify_if_paths_exist(
        [
            train_output_path,
            label_binarizer_path,
            test_output_path,
        ]
    ):
        return

    temporary_output_path = train_output_path + ".tmp"

    preprocess(input_path, temporary_output_path, text_cols, meta_cols)
    create_label_binarizer(temporary_output_path, label_binarizer_path)

    if test_output_path:
        split_data(
            temporary_output_path, train_output_path, test_output_path, test_split
        )
        shutil.rm(temporary_output_path)
    else:
        shutil.move(temporary_output_path, train_output_path)


if __name__ == "__main__":
    preprocess_wellcome_app()
