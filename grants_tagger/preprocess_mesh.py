"""
Preprocess JSON Mesh data from BioASQ to JSONL
"""
from configparser import ConfigParser
from pathlib import Path
import argparse
import configparser
import random
import json
import sys
import os
import typer
import yaml
import shutil

from tqdm import tqdm
from typing import List, Optional
import pandas as pd
import typer

from grants_tagger.utils import write_jsonl, verify_if_paths_exist
from grants_tagger.label_binarizer import create_label_binarizer
from grants_tagger.split_data import split_data


def yield_raw_data(input_path):
    with open(input_path, encoding="latin-1") as f_i:
        f_i.readline()  # skip first line ({"articles":[) which is not valid JSON
        for i, line in enumerate(f_i):
            item = json.loads(line[:-2])
            yield item


def process_data(item, filter_tags=None, filter_years=None):
    text = item["abstractText"]
    tags = item["meshMajor"]
    journal = item["journal"]
    year = item["year"]
    if filter_tags:
        tags = list(set(tags).intersection(filter_tags))
    if not tags:
        return
    if filter_years:
        min_year, max_year = filter_years.split(",")
        if year > int(max_year):
            return
        if year < int(min_year):
            return
    data = {"text": text, "tags": tags, "meta": {"journal": journal, "year": year}}
    return data


def yield_data(input_path, filter_tags, filter_years, buffer_size=10_000):
    data_batch = []
    for item in tqdm(yield_raw_data(input_path), total=15_000_000):  # approx 15M docs
        processed_item = process_data(item, filter_tags, filter_years)

        if processed_item:
            data_batch.append(processed_item)

        if len(data_batch) >= buffer_size:
            yield data_batch
            data_batch = []

    if data_batch:
        yield data_batch


def preprocess_mesh(
    raw_data_path,
    processed_data_path,
    mesh_tags_path=None,
    filter_years=None,
    n_max=None,
    buffer_size=10_000,
):
    if mesh_tags_path:
        filter_tags_data = pd.read_csv(mesh_tags_path)
        filter_tags = filter_tags_data["DescriptorName"].tolist()
        filter_tags = set(filter_tags)
    else:
        filter_tags = None

    if filter_years:
        min_year, max_year = filter_years.split(",")
        filter_years = [int(min_year), int(max_year)]

    # If only using a tiny set of data, need to reduce buffer size
    if n_max is not None and n_max < buffer_size:
        buffer_size = n_max

    with open(processed_data_path, "w") as f:
        for i, data_batch in enumerate(
            yield_data(
                raw_data_path, filter_tags, filter_years, buffer_size=buffer_size
            )
        ):
            write_jsonl(f, data_batch)
            if n_max and (i + 1) * buffer_size >= n_max:
                break


preprocess_mesh_app = typer.Typer()


@preprocess_mesh_app.command()
def preprocess_mesh_cli(
    input_path: Optional[str] = typer.Argument(None, help="path to BioASQ JSON data"),
    train_output_path: Optional[str] = typer.Argument(
        None, help="path to JSONL output file that will be generated for the train set"
    ),
    label_binarizer_path: Optional[Path] = typer.Argument(
        None, help="path to pickle file that will contain the label binarizer"
    ),
    test_output_path: Optional[str] = typer.Option(
        None, help="path to JSONL output file that will be generated for the test set"
    ),
    mesh_tags_path: Optional[str] = typer.Option(
        None, help="path to mesh tags to filter"
    ),
    test_split: Optional[float] = typer.Option(
        0.01, help="split percentage for test data. if None no split."
    ),
    filter_years: Optional[str] = typer.Option(
        None, help="years to keep in form min_year,max_year with both inclusive"
    ),
    config: Optional[Path] = typer.Option(
        None, help="path to config files that defines arguments"
    ),
    n_max: Optional[int] = typer.Option(
        None,
        help="Maximum limit on the number of datapoints in the set (including training and test)",
    ),
):

    params_path = os.path.join(os.path.dirname(__file__), "../params.yaml")
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # Default values from params
    if not mesh_tags_path:
        mesh_tags_path = params["preprocess_bioasq_mesh"].get("mesh_tags_path")

    # Default value from params
    if not filter_years:
        filter_years = params["preprocess_bioasq_mesh"].get("filter_years")

    if config:
        cfg = configparser.ConfigParser()
        cfg.read(config)

        input_path = cfg["preprocess"]["input"]
        train_output_path = cfg["preprocess"]["output"]
        mesh_tags_path = cfg["filter_mesh"].get("mesh_tags_path")
        test_split = cfg["preprocess"].getfloat("test_split")
        filter_years = cfg["preprocess"].get("filter_years")

    if verify_if_paths_exist(
        [
            train_output_path,
            label_binarizer_path,
            test_output_path,
        ]
    ):
        return

    temporary_output_path = train_output_path + ".tmp"
    preprocess_mesh(
        input_path,
        temporary_output_path,
        mesh_tags_path=mesh_tags_path,
        filter_years=filter_years,
        n_max=n_max,
    )
    create_label_binarizer(temporary_output_path, label_binarizer_path, sparse=True)

    if test_output_path:
        split_data(
            temporary_output_path, train_output_path, test_output_path, test_split
        )
        os.remove(temporary_output_path)
    else:
        shutil.move(temporary_output_path, train_output_path)


if __name__ == "__main__":
    typer.run(preprocess_mesh)
