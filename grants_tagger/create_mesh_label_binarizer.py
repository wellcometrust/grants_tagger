"""
Reads MeSH dataset and fits a Multilabel binarizer to 
the MeSH tags.
"""
from pathlib import Path
import argparse
import pickle
import json

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
    args = argparser.parse_args()

    create_label_binarizer(args.data, args.label_binarizer)
