"""
Preprocess JSON Mesh data from BioASQ to JSONL
"""
from configparser import ConfigParser
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse
import json
import os

import pandas as pd


def filter_disease_codes(mesh_descriptions_file):
    mesh_tree = ET.parse(mesh_descriptions_file)
    mesh_Df = pd.DataFrame(columns = ['DescriptorName', 'DescriptorUI', 'TreeNumberList'])

    for mesh in mesh_tree.getroot():
        try:
            # TreeNumberList e.g. A11.118.637.555.567.550.500.100
            mesh_tree = mesh[-2][0].text
            # DescriptorUI e.g. M000616943
            mesh_code = mesh[0].text
            # DescriptorName e.g. Mucosal-Associated Invariant T Cells
            mesh_name = mesh[1][0].text
        except IndexError:
            print("ERROR", file=sys.stderr)
        if mesh_tree.startswith('C') and not mesh_tree.startswith('C22') or mesh_tree.startswith('F03'):
            print(mesh_name)
            mesh_Df = mesh_Df.append({'DescriptorName':mesh_name, 'DescriptorUI':mesh_code, 'TreeNumberList':mesh_tree}, ignore_index=True)
#    mesh_Df.to_csv(mesh_export_file)


def yield_raw_data(input_path):
    with open(input_path, encoding='latin-1') as f_i:
        f_i.readline() # skip first line ({"articles":[) which is not valid JSON
        for i, line in enumerate(f_i):
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


def preprocess_mesh(input_path, output_path, filter_tags=None, mesh_metadata_path=None):
    if filter_tags == "disease":
        filter_tags_data = filter_disease_codes(mesh_metadata_path)
        filter_tags = filter_tags_data["DescriptorName"].tolist()
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

