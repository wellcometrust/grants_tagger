"""
Preprocess JSON Mesh data from BioASQ to JSONL
"""
from configparser import ConfigParser
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse
import random
import json
import sys
import os

from tqdm import tqdm
import pandas as pd


def filter_disease_codes(mesh_descriptions_file):
    mesh_tree = ET.parse(mesh_descriptions_file)
    mesh_Df = pd.DataFrame(columns = ['DescriptorName', 'DescriptorUI', 'TreeNumberList'])

    for mesh in tqdm(mesh_tree.getroot()):
        try:
            # TreeNumberList e.g. A11.118.637.555.567.550.500.100
            mesh_tree = mesh[-2][0].text
            # DescriptorUI e.g. M000616943
            mesh_code = mesh[0].text
            # DescriptorName e.g. Mucosal-Associated Invariant T Cells
            mesh_name = mesh[1][0].text
        except IndexError:
            # TODO: Add logger
            # print("ERROR", file=sys.stderr)
            pass
        if mesh_tree.startswith('C') and not mesh_tree.startswith('C22') or mesh_tree.startswith('F03'):
            mesh_Df = mesh_Df.append({'DescriptorName':mesh_name, 'DescriptorUI':mesh_code, 'TreeNumberList':mesh_tree}, ignore_index=True)
#    mesh_Df.to_csv(mesh_export_file)
    return mesh_Df

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

def yield_data(input_path, filter_tags, buffer_size=10_000):
    data_batch = []
    for item in tqdm(yield_raw_data(input_path), total=15_000_000): # approx 15M docs
        processed_item = process_data(item, filter_tags)
        if processed_item:
            data_batch.append(processed_item)

        if len(data_batch) >= buffer_size:
            yield data_batch
            data_batch = []
    
    if data_batch:
        yield data_batch

def yield_train_test_data(input_path, filter_tags, test_split=0.1, random_seed=42):
    for data_batch in yield_data(input_path, filter_tags):    
        random.seed(random_seed)
        random.shuffle(data_batch)
        n_test = int(test_split * len(data_batch))
        yield data_batch[:-n_test], data_batch[-n_test:]

def write_jsonl(f, data):
    for item in data:
        f.write(json.dumps(item))
        f.write("\n")

def preprocess_mesh(raw_data_path, processed_data_path, 
        test_split=None, filter_tags=None, mesh_metadata_path=None):
    if filter_tags == "disease":
        filter_tags_data = filter_disease_codes(mesh_metadata_path)
        filter_tags = filter_tags_data["DescriptorName"].tolist()
        filter_tags = set(filter_tags)
    else:
        filter_tags = None

    if test_split:
        data_dir, data_name = os.path.split(processed_data_path)
        processed_train_data_path = os.path.join(data_dir, "train_" + data_name) 
        processed_test_data_path = os.path.join(data_dir, "test_" + data_name)

        with open(processed_train_data_path, "w") as train_f, open(processed_test_data_path, "w") as test_f:
            for train_data_batch, test_data_batch in yield_train_test_data(raw_data_path, filter_tags, test_split):
                write_jsonl(train_f, train_data_batch)
                write_jsonl(test_f, test_data_batch)
    else:
        with open(processed_data_path, "w") as f:
            for data_batch in yield_data(raw_data_path, filter_tags):
                write_jsonl(f, data_batch)

