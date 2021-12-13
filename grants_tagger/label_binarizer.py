import argparse
import pickle

from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

from grants_tagger.utils import yield_tags


def create_label_binarizer(data_path, label_binarizer_path, sparse=False):
    """Creates, saves and returns a multilabel binarizer for targets Y"""
    label_binarizer = MultiLabelBinarizer(sparse_output=sparse)
    # TODO: pass Y_train here which can be generator or list
    label_binarizer.fit(yield_tags(data_path))

    with open(label_binarizer_path, "wb") as f:
        f.write(pickle.dumps(label_binarizer))

    return label_binarizer


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path", type=Path)
    argparser.add_argument("--label_binarizer_path", type=Path)
    argparser.add_argument("--sparse_labels", type=bool)

    args = argparser.parse_args()

    create_label_binarizer(
        args.data_path, args.label_binarizer_path, sparse_labels=args.sparse_labels
    )
