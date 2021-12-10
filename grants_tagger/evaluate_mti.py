"""
Evaluate MTI tool from NLM on MeSH subset. The subset is automatically tagged
by MTI through an email batch service and this scripts calculates performance
by comparing with the ground truth.
"""
from argparse import ArgumentParser
from pathlib import Path
import pickle
import json
import csv
import typer

from sklearn.metrics import classification_report, f1_score


def get_gold_tags(mesh_sample_data_path):
    gold_tags = []
    with open(mesh_sample_data_path) as f:
        for line in f:
            item = json.loads(line)
            gold_tags.append(item["tags"])
    return gold_tags


def get_mti_tags(mti_output_path, mesh_tags):
    mti_data = dict()
    with open(mti_output_path) as f:
        csvreader = csv.reader(f, delimiter="|")
        for line in csvreader:
            uid, tag, _, _ = line
            uid = int(uid)
            if uid not in mti_data:
                mti_data[uid] = []
            if tag in mesh_tags:
                mti_data[uid].append(tag)

    mti_tags = [mti_data[uid] for uid in range(len(mti_data))]
    return mti_tags


def evaluate_mti(
    label_binarizer_path, mesh_sample_data_path, mti_output_path, verbose=False
):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())
    disease_tags = set(label_binarizer.classes_)

    gold_tags = get_gold_tags(mesh_sample_data_path)
    mti_tags = get_mti_tags(mti_output_path, disease_tags)

    Y = label_binarizer.transform(gold_tags)
    Y_pred = label_binarizer.transform(mti_tags)

    f1 = f1_score(Y, Y_pred, average="micro")

    if verbose:
        print(classification_report(Y, Y_pred, target_names=label_binarizer.classes_))
    else:
        print(f1)

    return f1


evaluate_mti_app = typer.Typer()


@evaluate_mti_app.command()
def evaluate_mti_cli(
    data_path: Path = typer.Argument(..., help="path to sample JSONL mesh data"),
    label_binarizer_path: Path = typer.Argument(
        ..., help="path to pickled mesh label binarizer"
    ),
    mti_output_path: Path = typer.Argument(..., help="path to mti output txt"),
):

    evaluate_mti(label_binarizer_path, data_path, mti_output_path)


if __name__ == "__main__":
    evaluate_mti_app()
