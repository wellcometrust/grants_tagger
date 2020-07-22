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
        for i, line in enumerate(f_i):
            if i == 0:
                # skip first line ({"articles":[) which is JSON
                continue
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

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--input", type=Path, help="path to mesh JSON data")
    argparser.add_argument("--output", type=Path, help="path to output JSONL data")
    argparser.add_argument("--filter-tags", type=Path, help="path to txt file with tags to keep")
    argparser.add_argument("--config", type=Path, help="path to config files that defines arguments")
    args = argparser.parse_args()

    if args.config:
        cfg = ConfigParser()
        cfg.read(args.config)

        input_path = cfg["preprocess"]["input"]
        output_path = cfg["preprocess"]["output"]
        filter_tags_path = cfg["preprocess"]["filter_tags"]
    else:
        input_path = args.input
        output_path = args.output
        filter_tags_path = args.filter_tags

    if os.path.exists(output_path):
        print(f"{output_path} exists. Remove if you want to rerun.")
    else:
        preprocess_mesh(input_path, output_path, filter_tags_path)
