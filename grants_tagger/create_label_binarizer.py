# encoding: utf-8
"""
Creates multi label binarizer from JSONL with tags field
"""
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
import pickle
import json
import os

from sklearn.preprocessing import MultiLabelBinarizer


def yield_tags(data_path):
    """yields tags (list) line by line from file in data_path"""
    with open(data_path) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            yield item["tags"]


def create_label_binarizer(data_path, label_binarizer_path, sparse=False):
    data_tags = []
    for tags in yield_tags(data_path):
        data_tags.append(tags)
        
    label_binarizer = MultiLabelBinarizer(sparse_output=sparse)
    label_binarizer.fit(data_tags)
    with open(label_binarizer_path, 'wb') as f:
        pickle.dump(label_binarizer, f)


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--data', type=Path, help="path to processed JSONL with grants data", required=True)
    argparser.add_argument('--label_binarizer', type=Path, help="path to save label binarizer", required=True)
    argparser.add_argument('--sparse', type=bool, default=False, help="flag to output sparse labels in Y")
    argparser.add_argument('--config', type=Path, help="path to config that defines arguments")
    args = argparser.parse_args()

    if args.config:
        cfg = ConfigParser(allow_no_value=True)
        cfg.read(args.config)

        data_path = cfg["label_binarizer"]["data"]
        label_binarizer_path = cfg["label_binarizer"]["label_binarizer"]
        sparse = cfg["label_binarizer"].get("sparse", False)
    else:
        data_path = args.data
        label_binarizer_path = args.label_binarizer
        sparse = args.sparse

    if os.path.exists(label_binarizer_path):
        print(f"{label_binarizer_path} exists. Remove if you want to rerun.")
    else:
        create_label_binarizer(data_path, label_binarizer_path, sparse)
