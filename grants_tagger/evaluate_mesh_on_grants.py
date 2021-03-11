"""
Evaluate MeSH model on grants annotated with MeSH

The script works on an Excel file that contains 
columns that follow the pattern below
Disease MeSH#TAG_INDEX (ANNOTATOR)
"""
from argparse import ArgumentParser
from pathlib import Path
import pickle

from sklearn.metrics import f1_score
import pandas as pd

from grants_tagger.predict import predict


def get_tags(data, annotator):
    tag_columns = [col for col in data.columns if annotator in col]
    tags = []
    for _, row in data.iterrows():
        row_tags = [tag for tag in row[tag_columns] if not pd.isnull(tag)]
        tags.append(row_tags)
    return tags


def get_texts(data):
    texts = []
    for _, row in data.iterrows():
        text = row["Title"] + " " + row["Synopsis"]
        texts.append(text)
    return texts


def evaluate_mesh_on_grants(approach, data_path, model_path, label_binarizer_path):
    data = pd.read_excel(data_path, engine="openpyxl")

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    gold_tags = get_tags(data, "KW")
    Y = label_binarizer.transform(gold_tags)

    texts = get_texts(data)
    Y_pred = predict(texts, model_path, approach)
    f1 = f1_score(Y, Y_pred, average='micro')
    print(f"F1 micro is {f1}")

    unique_tags = len(set([t for tags in gold_tags for t in tags]))
    all_tags = len(label_binarizer.classes_)
    print(f"Gold dataset contains examples from {unique_tags} tags out of {all_tags}")


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--data_path', type=Path, help="path to validation data")
    argparser.add_argument('--model_path', type=Path, help="path to model")
    argparser.add_argument('--label_binarizer_path', type=Path, help="path to disease label binarizer")
    args = argparser.parse_args()

    evaluate_mesh_on_grants(args.data_path, args.model_path, args.label_binarizer_path)
