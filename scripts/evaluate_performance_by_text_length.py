"""
Evaluate model for science tags on different text lengths
to understand how performance changes by adding more text
and whether the first words are more important than taking
random words from the text
"""
from pathlib import Path
import argparse
import pickle
import random
import json

from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model_on_different_lengths(model, X_test, Y_test, random_sample=False):
    results = []
    for word_len in range(50, 300, 50):
        X_test_subset = []
        Y_test_subset = []
        for i in range(len(X_test)):
            x_test_words = X_test[i].split()
            if random_sample:
                random.seed(42)
                random.shuffle(x_test_words)
            X_test_subset.append(" ".join(x_test_words[:word_len]))
            Y_test_subset.append(Y_test[i, :])

        Y_test_subset = np.array(Y_test_subset)
        Y_pred = model.predict(X_test_subset)
        f1 = f1_score(Y_test_subset, Y_pred, average="micro")
        results.append((word_len, f1))

    return results

def evaluate_performance_by_text_length(data_path, label_binarizer_path, figure_path):
    with open("models/label_binarizer.pkl", "rb") as f:
        label_binarizer = pickle.loads(f.read())

    text = []
    tags = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            text.append(item["text"])
            tags.append(item["tags"])

    pipeline = Pipeline([
        ("vec", TfidfVectorizer(min_df=5, ngram_range=(1,2), stop_words="english")),
        ("svm", OneVsRestClassifier(SVC(kernel="linear", class_weight="balanced")))
    ])

    X = text
    Y = label_binarizer.transform(tags)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)
    pipeline.fit(X_train, Y_train)

    results = evaluate_model_on_different_lengths(pipeline, X_test, Y_test)
    results_random_sample = evaluate_model_on_different_lengths(pipeline, X_test, Y_test, random_sample=True)
    x, y = zip(*results)
    x_random, y_sample = zip(*results_random_sample)

    def shift_x(x, distance):
        return [x_+distance for x_ in x]

    _, ax = plt.subplots()
    ax.bar(shift_x(x, -10), y, width=20, label="first n words")
    ax.bar(shift_x(x, +10), y_sample, width=20, label="random n words")

    ax.set_xlabel("word count")
    ax.set_ylabel("f1")
    ax.set_title("Performance of science tagger by text length")
    ax.set_ylim(0.50)
    ax.legend()

    plt.savefig(figure_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__.strip())
    argparser.add_argument("--label_binarizer", type=Path)
    argparser.add_argument("--data_path", type=Path)
    argparser.add_argument("--figure_path", type=Path)
    args = argparser.parse_args()

    evaluate_performance_by_text_length(args.data_path, args.label_binarizer, args.figure_path)
