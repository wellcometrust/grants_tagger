"""
Adds tags based on SciBERT model to grants extracted from the warehouse
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

def tag_grants(grant, threshold):
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
    tags = predict_tags(grant['title'] + ' ' + grant['synopsis'], threshold=threshold)
    tagged_grant = {
        'Grant ID': grant['grant_id'],
        'Reference': grant['reference'],
        'Grant No.': grant['grant_no']
    }
    tagged_grant.update({
        f"Science Category#{i+1}": tag
        for i, tag in enumerate(tags)
    })
    return tagged_grant

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--grants', type=Path, help="path to grants csv")
    argparser.add_argument('--tagged_grants', type=Path, help="path to output csv")
    argparser.add_argument('--threshold', type=float, default=0.5, help="threshold upon which to assign tag")
    args = argparser.parse_args()
    
    with open(args.tagged_grants, 'w') as f_o:
        fieldnames = ["Grant ID", "Reference", "Grant No."]
        fieldnames += [f"Science Category#{i}" for i in range(1,8)]
        fieldnames += [f"Disease Category#{i}" for i in range(1,6)]
        csv_writer = csv.DictWriter(f_o, fieldnames=fieldnames)
        csv_writer.writeheader()
        for i, grant in enumerate(yield_grants(args.grants)):
            if i % 100 == 0:
                print(i)
            if i & 10 != 0:
                # keep only 1/10th
                continue
            tagged_grant = tag_grants(grant, args.threshold)
            csv_writer.writerow(tagged_grant)
