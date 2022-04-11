"""
Calculate stats of dataset in JSONL format
"""
from argparse import ArgumentParser
from pathlib import Path

import spacy
import json


def calculate_stats(data_path):
    nb_words = 0
    nlp = spacy.blank("en")
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            doc = nlp(item["text"])
            nb_words += len(doc)
    return {"vocab": len(doc.vocab), "words": nb_words}


argparser = ArgumentParser(description=__file__)
argparser.add_argument("--data", type=Path, help="JSONL dataset")
args = argparser.parse_args()

stats = calculate_stats(args.data)
print(stats)
