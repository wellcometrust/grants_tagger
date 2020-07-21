"""
Reads MeSH dataset and fits a Multilabel binarizer to 
the MeSH tags.
"""
from configparser import ConfigParser
from pathlib import Path
import argparse
import pickle
import json
import os

from sklearn.preprocessing import MultiLabelBinarizer


def yield_tags(data_path):
    """yields tags (list) line by line from file in data_path"""
    with open(data_path) as f:
        for i, line in enumerate(f):
#            print(i)
            item = json.loads(line)
            yield item["tags"]

def create_label_binarizer(data_path, label_binarizer_path):
    """reads file in data_path, creates label_binarizer and pickles it in label_binarizer_path"""
    mesh_tags = []
    for tags in yield_tags(data_path):
        mesh_tags.append(tags)
    
    label_binarizer = MultiLabelBinarizer(sparse_output=True)
    label_binarizer.fit(mesh_tags)

    with open(label_binarizer_path, 'wb') as f:
        pickle.dump(label_binarizer, f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--data", type=Path, help="path to read data with mesh tags")
    argparser.add_argument("--label_binarizer", type=Path, help="path to save multilabel binarizer")
    argparser.add_argument("--config", type=Path, help="path to config that defines arguments")
    args = argparser.parse_args()

    if args.config:
        config_parser = ConfigParser()
        data_path = cfg["label_binarizer"]["data"]
        label_binarizer_path = cfg["label_binarizer"]["label_binarizer"]
    else:
        data_path = args.data
        label_binarizer_path = args.label_binarizer

    if os.path.exists(label_binarizer_path):
        print(f"{label_binarizer_path} exists. Remove if you want to rerun.")
    else:
        create_label_binarizer(data_path, label_binarizer_path)
