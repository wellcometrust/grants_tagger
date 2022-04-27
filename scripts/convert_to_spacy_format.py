"""
Convert JSONL to SpaCy JSON format
"""
from argparse import ArgumentParser
from pathlib import Path
import json

from spacy.gold import docs_to_json
import spacy


def get_labels(data_path):
    labels = set()
    with open(data_path) as f_i:
        for i, line in enumerate(f_i):
            item = json.loads(line)
            labels.update(item["tags"])
    return labels


def yield_docs(data_path, labels):
    nlp = spacy.blank("en")
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    with open(data_path) as f_i:
        for i, line in enumerate(f_i):
            item = json.loads(line)
            doc = nlp(item["text"])
            doc.cats = {label: label in item["tags"] for label in labels}
            yield doc


def convert_to_spacy_format(data_path, output_path):
    labels = get_labels(args.data)
    with open(output_path, "w") as f_o:
        json_format = []
        for i, doc in enumerate(yield_docs(args.data, labels)):
            print(i)
            json_format.append(docs_to_json(doc, id=i))
        f_o.write(json.dumps(json_format))


if __name__ == "__main__":
    argparser = ArgumentParser(description=__file__)
    argparser.add_argument("--data", type=Path, help="textcat JSONL data")
    argparser.add_argument("--output", type=Path, help="output JSON format")
    args = argparser.parse_args()

    convert_to_spacy_format(args.data, args.output)
