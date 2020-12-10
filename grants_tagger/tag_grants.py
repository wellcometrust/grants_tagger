"""
Adds tags to grants based on the model for science or mesh
"""
from pathlib import Path
import argparse
import csv

from grants_tagger.predict import predict_tags


def yield_grants(grants_path):
    """yields grants row by row from file"""
    with open(grants_path) as f:
        csv_reader = csv.DictReader(f)
        for grant in csv_reader:
            yield grant


def yield_tagged_grants(grants, model_path, label_binarizer_path, threshold):
    """
    Tags grants and outputs tagged grant data structure

    Args:
        grant: dict with keys grant_id, reference, grant_no, title, synopsis
    Returns
        tagged_grant: dict with keys
            Grant ID, Reference, Grant No
            Tag #{1..10}
    """
    grants_texts = [g["title"]+g["synopsis"] for g in grants]
    tags = predict_tags(
        grants_texts,
        threshold=threshold,
        model_path=model_path,
        label_binarizer_path=label_binarizer_path
    )
    for grant, tags in zip(grants, grants_tags):
        tagged_grant = {
            'Grant ID': grant['grant_id'],
            'Reference': grant['reference'],
            'Grant No.': grant['grant_no']
        }
        tagged_grant.update({
            f"Tag #{i+1}": tag
            for i, tag in enumerate(tags)
            if i <= 10 
        })
        yield tagged_grant


def tag_grants(grants_path, tagged_grants_path, model_path, label_binarizer_path, threshold=0.5):
    with open(tagged_grants_path, 'w') as f_o:
        fieldnames = ["Grant ID", "Reference", "Grant No."]
        fieldnames += [f"Tag #{i}" for i in range(1,11)]
        csv_writer = csv.DictWriter(f_o, fieldnames=fieldnames)
        csv_writer.writeheader()

        grants = []
        for i, grant in enumerate(yield_grants(grants_path)):
            grants.append(grant)

            if len(grants) >= 512:
                for tagged_grant in yield_tagged_grants(grants, model_path, label_binarizer_path, threshold):
                    csv_writer.writerow(tagged_grant)

            grants = []


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--grants_path', type=Path, help="path to grants csv")
    argparser.add_argument('--tagged_grants_path', type=Path, help="path to output csv")
    argparser.add_argument("--model_path", type=Path, help="path to model")
    argparser.add_argument("--label_binarizer_path", type=Path, help="label binarizer for Y")
    argparser.add_argument('--threshold', type=float, default=0.5, help="threshold upon which to assign tag")
    args = argparser.parse_args()

    tag_grants(args.grants_path, args.tagged_grants_path, args.model_path,
               args.label_binarizer, args.threshold)
