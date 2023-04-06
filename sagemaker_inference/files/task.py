import tempfile
import pickle
import json
import csv
import logging
from argparse import ArgumentParser
import datetime
import sys
from flask import Flask, request
import flask
import csv
from grants_tagger.predict import predict_tags

csv.field_size_limit(sys.maxsize)


def yield_grants(input_file):
    """yield grants from s3 location line by line"""
    with open(input_file, "r") as tf:
        csv_reader = csv.DictReader(tf, delimiter=",", quotechar='"')
        for line in csv_reader:
            yield line


app = Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    # check if we can ping into the model
    prediction = predict_tags(
        [],
        "/opt/model",
        "/opt/model/vectorizer.pkl",
        probabilities=False,
        threshold=0.5,
        parameters=None,
        config=None,
    )

    # Check if the classifier was loaded correctly
    health = prediction is not None
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def tag_grants():
    input_json = flask.request.get_json()

    data = request.get_data().decode("utf-8")
    input_csv = csv.DictReader(data.splitlines())
    TG = TagGrants()
    output = TG(input_file=input_csv)
    return flask.Response(response=output, status=200, mimetype='application/json')


class TagGrants:
    """
    Task to read grants from file, tag and output
        to CSV file
    Args:
        input_file: Grants data source location
        output_file: Location to output data to
        model_path: The path to the model file
        label_binarizer_path: The path the binarizer pkl file
        cap: (default=None) a cap of number of records to process (testing)
    """

    def __init__(
        self,
        input_file,
        threshold,
        model_path,
        label_binarizer_path,
        cap,
    ):
        self.input_file = input_file
        self.threshold = threshold
        self.model_path = model_path
        self.label_binarizer_path = label_binarizer_path
        self.cap = cap

    def execute(self):
        output_data = []
        texts = []
        grant_ids = []
        grants_processed = 0
        for grant in yield_grants(self.input_file):
            text = " ".join(
                [
                    grant["Title"].replace("No Data Entered", ""),
                    grant["Synopsis"].replace("No Data Entered", ""),
                    grant["Lay Summary"].replace("No Data Entered", ""),
                    grant["Research Question"].replace("No Data Entered", ""),
                ]
            )
            texts.append(text)
            grant_ids.append(grant["Grant ID"])
            grants_processed += 1
            if self.cap is not None and grants_processed >= int(self.cap):
                break

        tags = predict_tags(
            texts,
            self.model_path,
            self.label_binarizer_path,
            threshold=self.threshold,
            probabilities=True,
        )

        for grant_id, grant_tags in zip(grant_ids, tags):
            grant_output = {"Grant ID": grant_id, "Tags": []}
            if grant_tags:
                for tag, prob in grant_tags.items():
                    grant_output["Tags"].append({"Tag": tag, "Prob": prob})
            if not grant_tags:
                grant_output["Tags"].append({"Tag": "no tag", "Prob": -1})
            output_data.append(grant_output)

        logging.info("Grants Tagging Complete: %s", self.output_file)
        return output_data


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()

    parser.add_argument("-i", "--input", default=None, required=True)
    parser.add_argument("-o", "--output", default=None, required=True)
    parser.add_argument("-t", "--threshold", default=1e-5, type=float, required=True)
    parser.add_argument("-m", "--model_path", default=None, required=True)
    parser.add_argument("-l", "--label_binarizer_path", default=None, required=True)
    parser.add_argument(
        "-c", "--cap", default=None, help="Cap the number of records that are processed"
    )

    args = vars(parser.parse_args())

    task = TagGrants(
        args.get("input"),
        args.get("output"),
        args.get("threshold"),
        args.get("model_path"),
        args.get("label_binarizer_path"),
        args.get("cap"),
    )
    task.execute()
