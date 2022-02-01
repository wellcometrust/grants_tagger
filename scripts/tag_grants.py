"""
Adds tags to grants based on the model for science or mesh
"""
from pathlib import Path
import math
import argparse
import csv
import sys

csv.field_size_limit(sys.maxsize)

from typing import List

import typer
import pandas as pd
from tqdm import tqdm

from grants_tagger.predict import predict_tags

app = typer.Typer()


def yield_batched_grants(input_file, batch_size=256, grant_text_fields=["title", "synopsis"], text_null_value="No Data Entered"):
    """yield grants in batches from input file line by line"""
    with open(input_file, "r") as tf:
        csv_reader = csv.DictReader(tf, delimiter=",", quotechar='"')
        lines = []
        for line in csv_reader:
            text = " ".join([line[field].replace(text_null_value, "").rstrip()
                    for field in grant_text_fields])
            if text:
                line["text"] = text
                lines.append(line)
            if len(lines) >= batch_size:
                yield lines
                lines = []
        if lines:
            yield lines

@app.command()
def tag_grants(grants_path, tagged_grants_path, model_path, label_binarizer_path, approach, threshold: float = 0.5,
        grant_id_field = "grant_id", grant_text_fields: List[str] = ["title", "synopsis"], text_null_value="No Data Entered"):
    
    n_grants = len(pd.read_csv(grants_path))

    with open(tagged_grants_path, "w") as tagged_grants_tf:
        fieldnames = ["Grant id", "Tag", "Prob"]
        csv_writer = csv.DictWriter(
            tagged_grants_tf, fieldnames=fieldnames, delimiter=",",
            quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writeheader()

        for grants in tqdm(yield_batched_grants(grants_path, 128, grant_text_fields, text_null_value),total=math.floor(n_grants/128)+1):
            grants_text = [grant['text'] for grant in grants]
            
            # Ignore the grants that don't have text.
            relevant_indices = [i for i, text in enumerate(grants_text) if text]

            grants = [grants[i] for i in relevant_indices]
            grants_text = [grants_text[i] for i in relevant_indices]

            if len(grants) > 0:
                # Removes consecutive white spaces which are uninformative and may cause error #30
                grants_tags = predict_tags(
                        grants_text,
                        model_path, 
                        label_binarizer_path,
                        approach=approach,
                        probabilities=True, 
                        threshold=threshold
                )
                
                for grant, tags in zip(grants, grants_tags):
                    for tag, prob in tags.items():
                        csv_writer.writerow({
                            'Grant id': grant[grant_id_field],
                            'Tag': tag,
                            'Prob': prob
                        })

                tagged_grants_tf.flush()


if __name__ == "__main__":
    app()
