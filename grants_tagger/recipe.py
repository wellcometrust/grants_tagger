# coding: utf8
from __future__ import unicode_literals
from functools import partial

import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.sorters import prefer_high_scores
from prodigy.util import split_string

import pickle
import os


class CustomModel(object):
    def __init__(self, label, model_path, label_binarizer_path):
        self.label = label
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(label_binarizer_path, "rb") as f:
            self.label_binarizer = pickle.load(f)

        self.valid_labels = self.label_binarizer.classes_
        assert self.label in self.valid_labels

    def __call__(self, stream):
        for eg in stream:
            eg["label"] = self.label

            label_idx = list(self.valid_labels).index(self.label)
            score = self.model.predict_proba([eg["text"]])[0][label_idx]

            eg["meta"] = eg.get("meta", {})
            eg["meta"]["score"] = score
            yield (score, eg)


@prodigy.recipe(
    "textcat.teach-custom-model",
    dataset=("The dataset to use", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    model=("Path to trained model", "positional", None, str),
    label_binarizer=("Path to label binarizer", "positional", None, str),
    label=("One label", "option", "l", str),
    goal=(
        "The number of positive examples that you aim to gather",
        "option",
        "100",
        int,
    ),
)
def textcat_custom_model(dataset, source, model, label_binarizer, label, goal):
    """
    Use active learning-powered text classification with a scikit model.
    """

    stream = JSONL(source)

    model = CustomModel(
        label=label, model_path=model, label_binarizer_path=label_binarizer
    )

    stream = prefer_high_scores(model(stream))

    total_accepted = 0

    def update(answers):
        nonlocal total_accepted
        accepted = [a for a in answers if a["answer"] == "accept"]
        total_accepted += len(accepted)

    def progress(*args, **kwargs):
        nonlocal goal
        return total_accepted / goal + 0.00001

    return {
        "view_id": "classification",
        "dataset": dataset,
        "stream": stream,
        "update": update,
        "progress": progress,
        "config": {"label": label},
    }
