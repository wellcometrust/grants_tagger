"""
Preprocess JSON Mesh data from BioASQ to JSONL
"""
from pathlib import Path
import argparse
import json

def yield_raw_data(input_path):
    with open(input_path, encoding='latin-1') as f_i:
        for line in f_i:
            item = json.loads(line[:-2])
            yield item

def process_data(item, filter_tags=None):
    text = item["abstractText"]
    tags = item["meshMajor"]
    if filter_tags:
        tags = list(set(tags).intersection(filter_tags))
    if not tags:
        return
    data = {
        "text": text,
        "tags": tags,
        "meta": {}
    }
    return data

def preprocess_mesh(input_path, output_path, filter_tags_path=None):
    if filter_tags_path:
        with open(filter_tags_path) as f_f:
            filter_tags = {tag.strip() for tag in f_f}
    else:
        filter_tags = None

    def yield_data(input_path, filter_tags):
        for item in yield_raw_data(input_path):
            processed_item = process_data(item, filter_tags)
            if processed_item:
                yield processed_item

    with open(output_path, 'w') as f_o:
        for data in yield_data(input_path, filter_tags):
            f_o.write(json.dumps(data))
            f_o.write("\n")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--input", type=Path, help="path to mesh JSON data")
    argparser.add_argument("--output", type=Path, help="path to output JSONL data")
    argparser.add_argument("--filter-tags", type=Path, help="path to txt file with tags to keep")
    args = argparser.parse_args()

    preprocess_mesh(args.input, args.output, args.filter_tags)
