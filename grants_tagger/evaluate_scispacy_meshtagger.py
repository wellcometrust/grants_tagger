"""
Evaluates SciSpacyMesh tagger against a subset of Mesh
"""
from argparse import ArgumentParser
import pickle
import typer

import pandas as pd
from pathlib import Path

from grants_tagger.scispacy_meshtagger import SciSpacyMeshTagger
from grants_tagger.utils import load_data


def evaluate_scispacy_meshtagger(
    mesh_label_binarizer_path, mesh_tags_path, mesh_data_path
):
    with open(mesh_label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    disease_mesh_data = pd.read_csv(mesh_tags_path)
    mesh_name2tag = {
        name: tag
        for tag, name in zip(
            disease_mesh_data["DescriptorUI"], disease_mesh_data["DescriptorName"]
        )
    }

    mesh_tags_names = label_binarizer.classes_
    mesh_tags = [mesh_name2tag[name] for name in mesh_tags_names]

    X, Y, _ = load_data(mesh_data_path, label_binarizer)

    scispacy_meshtagger = SciSpacyMeshTagger(mesh_tags)
    scispacy_meshtagger.fit()
    score = scispacy_meshtagger.score(X, Y)
    print(score)


evaluate_scispacy_app = typer.Typer()


@evaluate_scispacy_app.command()
def evaluate_scispacy_cli(
    data_path: Path = typer.Argument(
        ..., help="JSONL of mesh data that contains text, tags and meta per line"
    ),
    label_binarizer_path: Path = typer.Argument(
        ..., help="label binarizer that transforms mesh names to binarized format"
    ),
    mesh_metadata_path: Path = typer.Argument(
        ..., help="csv that contains metadata about mesh such as UI, Name etc"
    ),
):

    try:
        from grants_tagger.evaluate_scispacy_meshtagger import (
            evaluate_scispacy_meshtagger,
        )
    except ModuleNotFoundError:
        print(
            "Scispacy not installed. To use it install separately pip install -r requirements_scispacy.txt"
        )
    finally:
        evaluate_scispacy_meshtagger(
            label_binarizer_path, mesh_metadata_path, data_path
        )


if __name__ == "__main__":
    evaluate_scispacy_app()
