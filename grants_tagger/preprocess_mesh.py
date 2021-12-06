"""
Preprocess JSON Mesh data from BioASQ to JSONL
"""
from configparser import ConfigParser
from pathlib import Path
import argparse
import random
import json
import sys
import os

from tqdm import tqdm
import pandas as pd
from grants_tagger.utils import write_jsonl


def yield_raw_data(input_path):
    with open(input_path, encoding="latin-1") as f_i:
        f_i.readline()  # skip first line ({"articles":[) which is not valid JSON
        for i, line in enumerate(f_i):
            item = json.loads(line[:-2])
            yield item


def process_data(item, filter_tags=None):
    text = item["abstractText"]
    tags = item["meshMajor"]
    if filter_tags:
        tags = list(set(tags).intersection(filter_tags))
    if not tags:
        return
    data = {"text": text, "tags": tags, "meta": {}}
    return data


def yield_data(input_path, filter_tags, buffer_size=10_000):
    data_batch = []
    for item in tqdm(yield_raw_data(input_path), total=15_000_000):  # approx 15M docs
        processed_item = process_data(item, filter_tags)
        if processed_item:
            data_batch.append(processed_item)

        if len(data_batch) >= buffer_size:
            yield data_batch
            data_batch = []

    if data_batch:
        yield data_batch


def preprocess_mesh(raw_data_path, processed_data_path, mesh_tags_path=None):
    if mesh_tags_path:
        filter_tags_data = pd.read_csv(mesh_tags_path)
        filter_tags = filter_tags_data["DescriptorName"].tolist()
        filter_tags = set(filter_tags)
    else:
        filter_tags = None

    with open(processed_data_path, "w") as f:
        for data_batch in yield_data(raw_data_path, filter_tags):
            write_jsonl(f, data_batch)
