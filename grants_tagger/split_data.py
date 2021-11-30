import json
import pandas as pd
import os
import random
import argparse

from pathlib import Path

from functools import partial
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from grants_tagger.utils import write_jsonl


def load_data(data_path, label_binarizer=None, X_format="List"):
    """Load data from the dataset."""
    print("Loading data...")

    texts = []
    tags = []
    meta = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)

            texts.append(data["text"])
            tags.append(data["tags"])
            meta.append(data["meta"])

    if label_binarizer:
        tags = label_binarizer.transform(tags)

    if X_format == "DataFrame":
        X = pd.DataFrame(meta)
        X["text"] = texts
        return X, tags, meta

    return texts, tags, meta


def yield_texts(data_path):
    """Yields texts from JSONL with text field"""
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            yield item["text"]


def yield_tags(data_path, label_binarizer=None):
    """Yields tags from JSONL with tags field. Transforms if label binarizer provided."""
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)

            if label_binarizer:
                # TODO: Make more efficient by using a buffer
                yield label_binarizer.transform([item["tags"]])[0]
            else:
                yield item["tags"]


def load_train_test_data(
    train_data_path,
    label_binarizer,
    test_data_path=None,
    test_size=None,
    data_format="list",
):
    """
    train_data_path: path. path to JSONL data that contains text and tags fields
    label_binarizer: MultiLabelBinarizer. multilabel binarizer instance used to transform tags
    test_data_path: path, default None. path to test JSONL data similar to train_data
    test_size: float, default None. if test_data_path not provided, dictates portion to be used as test
    data_format: str, default list. controls data are returned as lists or generators for memory efficiency
    """
    if data_format == "list":
        if test_data_path:
            X_train, Y_train, _ = load_data(train_data_path, label_binarizer)
            X_test, Y_test, _ = load_data(test_data_path, label_binarizer)

        else:
            X, Y, _ = load_data(train_data_path, label_binarizer)
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, random_state=42, test_size=test_size
            )
    else:
        if test_data_path:
            X_train = partial(yield_texts, train_data_path)
            Y_train = partial(yield_tags, train_data_path, label_binarizer)
            X_test = partial(yield_texts, test_data_path)
            Y_test = partial(yield_tags, test_data_path, label_binarizer)
        else:
            # note that we do not split the data, we assume the user has them splitted
            X_train = partial(yield_texts, train_data_path)
            Y_train = partial(yield_tags, train_data_path, label_binarizer)
            X_test = None
            Y_test = None

    return X_train, X_test, Y_train, Y_test


def yield_data_batch(input_path, buffer_size=10_000):
    data_batch = []
    with open(input_path, encoding="latin-1") as f_i:
        for i, line in enumerate(f_i):
            item = json.loads(line)
            data_batch.append(item)

            if len(data_batch) >= buffer_size:
                yield data_batch
                data_batch = []

        if data_batch:
            yield data_batch


def yield_train_test_data(input_path, test_split=0.1, random_seed=42):
    for data_batch in yield_data_batch(input_path):
        random.seed(random_seed)
        random.shuffle(data_batch)
        n_test = int(test_split * len(data_batch))
        yield data_batch[:-n_test], data_batch[-n_test:]


def split_data(
    processed_data_path,
    processed_train_data_path,
    processed_test_data_path,
    test_split=None,
):
    """
    processed_data_path: path. path to JSONL data that contains processed data
    processed_train_data_path: path. path to JSONL data that contains processed train data
    processed_test_data_path: path. path to JSONL data that contains processed test data
    test_split: float, default None. dictates portion to be used as test
    """

    with open(processed_train_data_path, "w") as train_f, open(
        processed_test_data_path, "w"
    ) as test_f:
        for train_data_batch, test_data_batch in yield_train_test_data(
            processed_data_path, test_split
        ):
            write_jsonl(train_f, train_data_batch)
            write_jsonl(test_f, test_data_batch)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=Path)
    argparser.add_argument("--train_data_path", type=Path)
    argparser.add_argument("--test_data_path", type=Path)
    argparser.add_argument("--test_split", type=float)
    args = argparser.parse_args()

    split_data(
        args.data_path,
        args.train_data_path,
        args.test_data_path,
        args.test_split,
    )
