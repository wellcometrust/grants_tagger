"""
Evaluates SciSpacyMesh tagger against a subset of Mesh
"""
from argparse import ArgumentParser
import pickle

import pandas as pd

from grants_tagger.scispacy_meshtagger import SciSpacyMeshTagger
from grants_tagger.utils import load_data


def evaluate_scispacy_meshtagger(mesh_label_binarizer_path, mesh_tags_path, mesh_data_path):
    with open(mesh_label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    disease_mesh_data = pd.read_csv(mesh_tags_path)
    mesh_name2tag = {name: tag for tag, name in zip(disease_mesh_data['DescriptorUI'],
                                                    disease_mesh_data['DescriptorName'])}

    mesh_tags_names = label_binarizer.classes_
    mesh_tags = [mesh_name2tag[name] for name in mesh_tags_names]

    X, Y, _ = load_data(mesh_data_path, label_binarizer)

    scispacy_meshtagger = SciSpacyMeshTagger(mesh_tags)
    scispacy_meshtagger.fit()
    score = scispacy_meshtagger.score(X, Y)
    print(score)


if __name__ == "__main__":
    argparser = ArgumentParser(description=__doc__.strip())
    argparser.add_argument("--mesh_label_binarizer", help="label binarizer that transforms mesh names to binarized format")
    argparser.add_argument("--mesh_tags", help="csv that contains metadata about mesh such as UI, Name etc")
    argparser.add_argument("--mesh_data", help="JSONL of mesh data that contains text, tags and meta per line")
    args = argparser.parse_args()

    evaluate_scispacy_meshtagger(args.mesh_label_binarizer, args.mesh_tags, args.mesh_data)
