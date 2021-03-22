"""
Adds tags to grants based on the model for science or mesh
"""
from pathlib import Path
import argparse
import csv

from grants_tagger.predict import predict_tags


def yield_grants(grants_path, batch_size=32):
    """yields grants in batches from file"""
    with open(grants_path) as f:
        csv_reader = csv.DictReader(f)

        grants = []
        for grant in csv_reader:
            grants.append(grant)

            if len(grants) >= batch_size:
                yield grants
                grants = []

        if grants:
            yield grants


def yield_tagged_grants(grants, model_path, label_binarizer_path, approach, threshold):
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
    grants_tags = predict_tags(
        grants_texts,
        threshold=threshold,
        approach=approach,
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
            if i < 10 
        })
        yield tagged_grant


def tag_grants(grants_path, tagged_grants_path, model_path, label_binarizer_path, approach, threshold=0.5):
    with open(tagged_grants_path, 'w') as f_o:
        fieldnames = ["Grant ID", "Reference", "Grant No."]
        fieldnames += [f"Tag #{i}" for i in range(1,11)]
        csv_writer = csv.DictWriter(f_o, fieldnames=fieldnames)
        csv_writer.writeheader()

        for grants in yield_grants(grants_path, batch_size=512):
            for tagged_grant in yield_tagged_grants(grants, model_path, label_binarizer_path, approach, threshold):
                csv_writer.writerow(tagged_grant)
