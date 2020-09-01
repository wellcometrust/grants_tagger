"""
Tag grants with MeSH tags
"""
from argparse import ArgumentParser
from pathlib import Path
import csv

from grants_tagger.predict_mesh import predict_mesh_tags as predict_tags

DEFAULT_MODEL_PATH = "models/disease_mesh_tfidf-svm-2020.07.0/"
DEFAULT_LABEL_BINARIZER_PATH = "models/disease_mesh_label_binarizer.pkl"

def yield_grants(grants_path):
    """yields grants row by row from file"""
    with open(grants_path) as f:
        csv_reader = csv.DictReader(f)
        for grant in csv_reader:
            yield grant

def yield_tagged_grants(grants, tags):
    """
    Tags grants and outputs tagged grant data structure

    Args:
        grant: dict with keys grant_id, reference, grant_no, title, synopsis
    Returns
        tagged_grant: dict with keys
            Grant ID, Reference, Grant No
            ScienceCategory#{1..9}
            DiseaseCategory#{1..6)
    """
    for grant, tags in zip(grants, tags):
        tagged_grant = {
            'Grant ID': grant['grant_id'],
            'Reference': grant['reference'],
            'Grant No.': grant['grant_no']
        }
        tagged_grant.update({
            f"Disease Category#{i+1}": tag
            for i, tag in enumerate(tags)
            if i < 5
        })
        yield tagged_grant

def tag_grants_with_mesh(grants_path, tagged_grants_path, model_path, label_binarizer_path):
    grants = [g for g in yield_grants(grants_path)]
    grants_text = [grant['title'] + ' ' + grant['synopsis'] for grant in grants]
    tags = predict_tags(grants_text, model_path=model_path, label_binarizer_path=label_binarizer_path)
    
    with open(tagged_grants_path, 'w') as f_o:
        fieldnames = ["Grant ID", "Reference", "Grant No."]
        fieldnames += [f"Science Category#{i}" for i in range(1,8)]
        fieldnames += [f"Disease Category#{i}" for i in range(1,6)]
        csv_writer = csv.DictWriter(f_o, fieldnames=fieldnames)
        csv_writer.writeheader()
        for i, tagged_grant in enumerate(yield_tagged_grants(grants, tags)):
            if i % 100 == 0:
                print(i)
            csv_writer.writerow(tagged_grant)

if __name__ == '__main__':
    argparser = ArgumentParser(description=__file__)
    argparser.add_argument('--grants', type=Path, help="")
    argparser.add_argument('--tagged_grants', type=Path, help="")
    argparser.add_argument('--model_path', type=Path, default=DEFAULT_MODEL_PATH, help="")
    argparser.add_argument('--label_binarizer_path', type=Path, default=DEFAULT_LABEL_BINARIZER_PATH, help="")
    args = argparser.parse_args()

    tag_grants_with_mesh(args.grants, args.tagged_grants, args.model_path, args.label_binarizer_path)
