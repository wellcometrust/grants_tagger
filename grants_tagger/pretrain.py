"""
Pretrain a model on unlabeled data to improve feature extraction
"""
from argparse import ArgumentParser
from pathlib import Path
from configparser import ConfigParser
import pickle
import os

from wellcomeml.ml import Doc2VecVectorizer
import pandas as pd

def pretrain(data_path, model_path, model_name):
    # TODO: Convert that to assume a JSONL with text field
    data = pd.read_csv(data_path)
    X = data["synopsis"].dropna().drop_duplicates()

    if model_name == "doc2vec":
        model = Doc2VecVectorizer(
            min_count=1,
            window_size=9,
            vector_size=300,
            negative=10,
            sample=1e-4,
            epochs=5
        )
    else:
        raise NotImplementedError
    model.fit(X)

    model.save(model_path)

if __name__ == "__main__":
    argparser = ArgumentParser(description=__doc__.strip())
    argparser.add_argument("--data_path", type=Path, help="data to pretrain model on")
    argparser.add_argument("--model_path", type=str, help="path to save model")
    argparser.add_argument("--model_name", type=str, help="name of model to pretrain")
    argparser.add_argument("--config", type=Path, help="config file with arguments for pretrain")
    args = argparser.parse_args()

    if args.config:
        cfg = ConfigParser(allow_no_value=True)
        cfg.read(args.config)

        try:
            cfg_pretrain = cfg["pretrain"]
        except KeyError:
            cfg_pretrain = {}
        data_path = cfg_pretrain.get("data_path")
        model_path = cfg_pretrain.get("model_path")
        model_name = cfg_pretrain.get("model_name")
    else:
        data_path = args.data_path
        model_path = args.model_path
        model_name = args.model_name

    if not model_path:
        print(f"No pretraining defined. Skipping.")
    elif os.path.exists(model_path):
        print(f"{model_path} exists. Remove if you want to rerun.")
    else:
        pretrain(data_path, model_path, model_name)
