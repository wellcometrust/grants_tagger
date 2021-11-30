import json
import pandas as pd
import os
import random
import argparse

from pathlib import Path

from grants_tagger.utils import write_jsonl


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
