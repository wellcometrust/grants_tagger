"""
Samples from processed mesh data in JSONL format 10k lines
that can be used for evaluating performance of other tools
like SciSpacy
"""
from argparse import ArgumentParser
from pathlib import Path
import random


RANDOM_SEED = 42
SAMPLE_SIZE = 10_000

def count_lines(jsonl_path):
    counter = 0
    with open(jsonl_path) as f:
        for line in f:
            counter += 1
    return counter

def yield_sample_data(data_path, sample_size=SAMPLE_SIZE, random_seed=RANDOM_SEED):
    lines_count = count_lines(data_path)

    line_indices = list(range(lines_count))
    random.seed(random_seed)
    random.shuffle(line_indices)
    sample_line_indices = set(line_indices[:sample_size])

    with open(data_path) as f:
        for i, line in enumerate(f):
            if i in sample_line_indices:
                yield line

def create_mesh_sample(mesh_data_path, sample_mesh_data_path, mti_input_path):
    if mti_input_path:
        mti_f = open(mti_input_path, "w")

    with open(sample_mesh_data_path, "w") as f:
        for i, item in enumerate(yield_sample_data(mesh_data_path)):
            f.write(item)

            if mti_input_path:
                uid = f"{i:08d}"
                text = item["text"].encode("ascii", errors="ignore")
                mti_f.write(f"{uid}|{text}\n")

    if mti_input_path:
        mti_f.close()

argparser = ArgumentParser(description=__doc__.strip())
argparser.add_argument("--mesh_data_path", type=Path, help="path to JSONL of processed mesh data")
argparser.add_argument("--sample_mesh_data_path", type=Path, help="path to output sample JSONL of mesh data")
argparser.add_argument("--mti_input_path", type=Path, default=None, help="path to output txt for MTI to use an input")
args = argparser.parse_args()

create_mesh_sample(args.mesh_data_path, args.sample_mesh_data_path, args.mti_input_path)
