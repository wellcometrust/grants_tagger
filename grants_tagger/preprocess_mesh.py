"""
Preprocess JSON Mesh data from BioASQ to JSONL
"""
from configparser import ConfigParser
from pathlib import Path
import argparse
import json
import os

import pandas as pd

def yield_raw_data(input_path):
    with open(input_path, encoding='latin-1') as f_i:
        f_i.readline() # skip first line ({"articles":[) which is not valid JSON
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
    data = {
        "text": text,
        "tags": tags,
        "meta": {}
    }
    return data


def preprocess_mesh(input_path, output_path, filter_tags_path=None):
    if filter_tags_path:
        filter_tags_data = pd.read_csv(filter_tags_path)
        filter_tags = filter_tags_data["DescriptorName"].tolist()
    else:
        filter_tags = None

    def yield_data(input_path, filter_tags):
        for item in yield_raw_data(input_path):
            processed_item = process_data(item, filter_tags)
            if processed_item:
                yield processed_item

    with open(output_path, 'w') as f_o:
        for data in yield_data(input_path, filter_tags):
            f_o.write(json.dumps(data))
            f_o.write("\n")

