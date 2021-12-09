"""
Evaluate human accuracy if you have annotated some data with humans
"""
import pickle
import typer

from pathlib import Path
from sklearn.metrics import classification_report, f1_score

from grants_tagger.utils import load_data


def evaluate_human(data_path, label_binarizer_path):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.load(f)

    X, Y, meta = load_data(data_path, label_binarizer)

    Y_tagger = label_binarizer.transform([m["Tagger1_tags"] for m in meta])

    human_f1 = f1_score(Y, Y_tagger, average="micro")

    report = classification_report(Y, Y_tagger)
    print(report)

    return human_f1


evaluate_human_app = typer.Typer()


@evaluate_human_app.command()
def evaluate_human_cli(data_path: Path, label_binarizer_path: Path):
    evaluate_human(data_path, label_binarizer_path)


if __name__ == "__main__":
    evaluate_human_app()
