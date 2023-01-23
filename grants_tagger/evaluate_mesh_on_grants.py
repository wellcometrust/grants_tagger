"""
Evaluate MeSH model on grants annotated with MeSH

The script works on an Excel file that contains
columns that follow the pattern below
Disease MeSH#TAG_INDEX (ANNOTATOR)
"""
from argparse import ArgumentParser
from pathlib import Path
import configparser
import pickle
import json
import typer

from typing import List, Optional
from sklearn.metrics import f1_score
import pandas as pd

from grants_tagger.models.create_model_xlinear import load_model


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


def evaluate_mesh_on_grants(
    approach,
    data_path,
    model_path,
    label_binarizer_path,
    results_path="mesh_on_grants_results.json",
    mesh_tags_path=None,
    parameters=None,
):
    data = pd.read_excel(data_path, engine="openpyxl")

    if mesh_tags_path:
        mesh_tags = pd.read_csv(mesh_tags_path)
        mesh_tags = set(mesh_tags["DescriptorName"].tolist())

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    if mesh_tags_path:
        mesh_tags_idx = [
            idx
            for idx, mesh_tag in enumerate(label_binarizer.classes_)
            if mesh_tag in mesh_tags
        ]

    gold_tags = get_tags(data, "KW")
    Y = label_binarizer.transform(gold_tags)

    texts = get_texts(data)

    model = load_model(approach, model_path, parameters=parameters)
    if getattr(model, "threshold") is None:
        model.threshold = 0.5

    Y_pred = model.predict(texts)

    if mesh_tags_path:
        Y = Y[:, mesh_tags_idx]
        Y_pred = Y_pred[:, mesh_tags_idx]

    f1 = f1_score(Y, Y_pred, average="micro")
    print(f"F1 micro is {f1}")

    with open(results_path, "w") as f:
        results = {"f1": f1}
        f.write(json.dumps(results))
    unique_tags = len(set([t for tags in gold_tags for t in tags]))
    all_tags = len(label_binarizer.classes_)
    print(f"Gold dataset contains examples from {unique_tags} tags out of {all_tags}")


evaluate_mesh_on_grants_app = typer.Typer()


@evaluate_mesh_on_grants_app.command()
def evaluate_mesh_on_grants_cli(
    approach: str = typer.Argument(..., help="model approach e.g.mesh-cnn"),
    model_path: str = typer.Argument(
        ..., help="comma separated paths to pretrained models"
    ),
    data_path: Path = typer.Argument(
        ..., help="path to data that was used for training"
    ),
    label_binarizer_path: Path = typer.Argument(..., help="path to label binarize"),
    threshold: Optional[str] = typer.Option(
        "0.5", help="threshold or comma separated thresholds used to assign tags"
    ),
    results_path: Optional[str] = typer.Option(None, help="path to save results"),
    mesh_tags_path: str = typer.Option(None, help="path to mesh subset to evaluate"),
    parameters: bool = typer.Option(
        None, help="stringified parameters for model evaluation, if any"
    ),
    config: Optional[Path] = typer.Option(
        None, help="path to config file that defines arguments"
    ),
):

    if config:
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)

        approach = cfg["ensemble"]["approach"]
        model_path = cfg["ensemble"]["models"]
        data_path = cfg["ensemble"]["data"]
        label_binarizer_path = cfg["ensemble"]["label_binarizer"]
        threshold = cfg["ensemble"]["threshold"]
        results_path = cfg["ensemble"].get("results_path", "results.json")

    if "," in threshold:
        threshold = [float(t) for t in threshold.split(",")]
    else:
        threshold = float(threshold)

    evaluate_mesh_on_grants(
        approach,
        data_path,
        model_path,
        label_binarizer_path,
        results_path=results_path,
        mesh_tags_path=mesh_tags_path,
        parameters=parameters,
    )


if __name__ == "__main__":
    evaluate_mesh_on_grants_app()
