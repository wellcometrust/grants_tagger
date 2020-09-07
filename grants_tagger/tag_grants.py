"""
Adds tags based on SciBERT model to grants extracted from the warehouse
"""
from pathlib import Path
import argparse
import csv
import os

from grants_tagger.predict import predict_tags


FILEPATH = os.path.dirname(__file__)
DEFAULT_SCIBERT_PATH = os.path.join(FILEPATH, '../models/scibert-2020.05.5')
DEFAULT_TFIDF_SVM_PATH = os.path.join(FILEPATH, '../models/tfidf-svm-2020.05.2.pkl')
DEFAULT_LABELBINARIZER_PATH = os.path.join(FILEPATH, '../models/label_binarizer.pkl')


def yield_grants(grants_path):
    """yields grants row by row from file"""
    with open(grants_path) as f:
        csv_reader = csv.DictReader(f)
        for grant in csv_reader:
            yield grant

def yield_tagged_grants(grants, scibert_path, tfidf_svm_path, label_binarizer_path, threshold):
    """
    Tags grants and outputs tagged grant data structure

    Args:
        grant: dict with keys grant_id, reference, grant_no, title, synopsis
    Returns
        tagged_grant: dict with keys
            Grant ID, Reference, Grant No
            ScienceCategory#{1..9}
            DiseaseCategory#{1..9)
    """
    grants_texts = [g["title"]+g["synopsis"] for g in grants]
    grants_tags = predict_tags(
        grants_texts,
        threshold=threshold,
        scibert_path=scibert_path,
        tfidf_svm_path=tfidf_svm_path,
        label_binarizer_path=label_binarizer_path
    )
    for grant, tags in zip(grants, grants_tags):
        tagged_grant = {
            'Grant ID': grant['grant_id'],
            'Reference': grant['reference'],
            'Grant No.': grant['grant_no']
        }
        tagged_grant.update({
            f"Science Category#{i+1}": tag
            for i, tag in enumerate(tags)
            if i < 7
        })
        yield tagged_grant

def tag_grants(grants_path, tagged_grants_path, scibert_path, tfidf_svm_path, label_binarizer_path, threshold=0.5):
    with open(tagged_grants_path, 'w') as f_o:
        fieldnames = ["Grant ID", "Reference", "Grant No."]
        fieldnames += [f"Science Category#{i}" for i in range(1,8)]
        fieldnames += [f"Disease Category#{i}" for i in range(1,6)]
        csv_writer = csv.DictWriter(f_o, fieldnames=fieldnames)
        csv_writer.writeheader()

        grants = []
        for i, grant in enumerate(yield_grants(grants_path)):
            if i % 10 != 0:
                # keep only 1/10th
                continue
            grants.append(grant)
        
        for tagged_grant in yield_tagged_grants(grants, scibert_path, tfidf_svm_path, label_binarizer_path, threshold):
            csv_writer.writerow(tagged_grant)
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--grants', type=Path, help="path to grants csv")
    argparser.add_argument('--tagged_grants', type=Path, help="path to output csv")
    argparser.add_argument("--scibert", type=Path, default=DEFAULT_SCIBERT_PATH, help="path to scibert model")
    argparser.add_argument("--tfidf_svm", type=Path, default=DEFAULT_TFIDF_SVM_PATH, help="path to scibert model")
    argparser.add_argument("--label_binarizer", type=Path, default=DEFAULT_LABELBINARIZER_PATH, help="label binarizer for Y")
    argparser.add_argument('--threshold', type=float, default=0.5, help="threshold upon which to assign tag")
    args = argparser.parse_args()

    tag_grants(args.grants, args.tagged_grants, args.scibert, args.tfidf_svm, args.label_binarizer, args.threshold)
