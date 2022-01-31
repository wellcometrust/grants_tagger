"""
Adds tags to grants based on the model for science or mesh
"""
from pathlib import Path
import argparse
import csv
from typing import List

import typer

from grants_tagger.predict import predict_tags

app = typer.Typer()


def yield_batched_grants(input_file, batch_size=256):
    """yield grants in batches from input file line by line"""
    with open(input_file, "r") as tf:
        csv_reader = csv.DictReader(tf, delimiter=",", quotechar='"')
        lines = []
        for line in csv_reader:
            lines.append(line)
            if len(lines) >= batch_size:
                yield lines
                lines = []
        if lines:
            yield lines

@app.command()
def tag_grants(grants_path, tagged_grants_path, model_path, label_binarizer_path, approach, threshold=0.5,
        grant_id_field = "grant_id", grant_text_fields: List[str] = ["title", "synopsis"], text_null_value="No Data Entered"):

    with open(tagged_grants_path, "w") as tagged_grants_tf:
        fieldnames = ["Grant id", "Tag", "Prob"]
        csv_writer = csv.DictWriter(
            tagged_grants_tf, fieldnames=fieldnames, delimiter=",",
            quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        csv_writer.writeheader()

        for grants in yield_batched_grants(grants_path, 128):
            grants_text = [
                " ".join([
                    grant[field].replace(text_null_value, "")
                    for field in grant_text_fields
                ])
                for grant in grants
            ]

            # Removes consecutive white spaces which are uninformative and may cause error #30
            grants_text = [" ".join(text.split()) for text in grants_text if text.strip()] # Removes empty text
            grants_tags = predict_tags(grants_text, model_path, label_binarizer_path, approach,
                    probabilities=True, threshold=threshold)

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
