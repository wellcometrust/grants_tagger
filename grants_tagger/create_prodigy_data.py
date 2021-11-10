"""
Convert JSON processed data to prodigy format to either
train with prodigy or teach i.e. annotate using a model
"""

from argparse import ArgumentParser
import pickle
import json

from grants_tagger.utils import load_data


def yield_chunks(text, labels, accept_labels, grant_id, mode):
    if mode == "train":
        for label in labels:
            yield json.dumps(
                {
                    "text": text,
                    "label": label,
                    "answer": "accept" if label in accept_labels else "reject",
                }
            )
    else:
        yield json.dumps({"text": text, "meta": {"grant_id": grant_id}})


def create_prodigy_data(data_path, label_binarizer_path, output_path, mode):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.load(f)

    texts, cats, meta = load_data(data_path)
    labels = label_binarizer.classes_

    with open(output_path, "w") as f:

        for text, accept_labels, meta_ in zip(texts, cats, meta):
            grant_id = meta_["Grant_ID"]
            for chunk in yield_chunks(text, labels, accept_labels, grant_id, mode):
                f.write(chunk)
                f.write("\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data", help="path to input JSON with grant data")
    parser.add_argument("--output", help="path for output data being produced")
    parser.add_argument("--mode", default="train", help="teach or train")
    parser.add_argument(
        "--label_binarizer", type=str, help="label binarizer for binarizing tags"
    )

    args = parser.parse_args()
    mode = args.mode
    assert mode in ["teach", "train"]

    create_prodigy_data(args.data, args.label_binarizer, args.output, args.mode)
