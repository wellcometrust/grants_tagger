# encoding: utf-8
"""
Creates multi label binarizer from JSONL with tags field
"""
from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path
import pickle
import os

from sklearn.preprocessing import MultiLabelBinarizer

from science_tagger.utils import load_data

def create_label_binarizer(data_path, label_binarizer_path):
    _, tags, _ = load_data(data_path)

    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit(tags)
    with open(label_binarizer_path, 'wb') as f:
        pickle.dump(label_binarizer, f)

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--data', type=Path, help="path to processed JSONL with grants data")
    argparser.add_argument('--label_binarizer', type=Path, help="path to save label binarizer")
    argparser.add_argument('--config', type=Path, help="path to config that defines arguments")
    args = argparser.parse_args()

    if args.config:
        cfg = ConfigParser()
        cfg.read(args.config)

        data_path = cfg["label_binarizer"]["data"]
        label_binarizer_path = cfg["label_binarizer"]["label_binarizer"]
    else:
        data_path = args.data
        label_binarizer_path = args.label_binarizer

    if os.path.exists(label_binarizer_path):
        print(f"{label_binarizer_path} exists. Remove if you want to rerun.")
    else:
        create_label_binarizer(data_path, label_binarizer_path)
