"""
Evaluate model performance on test set
"""
import pickle
import json
import configparser
import typer

from typing import List, Optional
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, classification_report
from wasabi import table, row
import scipy.sparse as sp

from grants_tagger.utils import load_train_test_data, load_data
from grants_tagger.models.create_model_xlinear import load_model


def predict_sparse_probs(model, X_test, batch_size=256, cutoff_prob=0.01):
    Y_pred_proba = []
    for i in range(0, X_test.shape[0], batch_size):
        Y_pred_proba_batch = model.predict_proba(X_test)
        Y_pred_proba_batch[Y_pred_proba_batch < cutoff_prob] = 0
        Y_pred_proba_batch = sp.csr_matrix(Y_pred_proba_batch)
        Y_pred_proba.append(Y_pred_proba_batch)
    Y_pred_proba = sp.vstack(Y_pred_proba_batch)
    return Y_pred_proba


def evaluate_model(
    approach,
    model_path,
    data_path,
    label_binarizer_path,
    threshold,
    split_data=True,
    results_path=None,
    full_report_path=None,
    sparse_y=False,
    parameters=None,
):
    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    if split_data:
        print(
            "Warning: Data will be split in the same way as train."
            " If you don't want that you set split_data=False"
        )
        _, X_test, _, Y_test = load_train_test_data(data_path, label_binarizer)
    else:
        X_test, Y_test, _ = load_data(data_path, label_binarizer)

    # Some models (e.g. MeshXLinear) need to know the parameters beforehand, to know which
    # Load function to use

    model = load_model(approach, model_path, parameters=parameters)

    if sparse_y:
        predict_sparse_probs(model, X_test)
    else:
        Y_pred_proba = model.predict_proba(X_test)

    if type(threshold) != list:
        threshold = [threshold]

    widths = (12, 5, 5, 5)
    header = ["Threshold", "P", "R", "F1"]
    print(table([], header, divider=True, widths=widths))

    results = []
    for th in threshold:
        Y_pred = Y_pred_proba > th
        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average="micro")
        full_report = classification_report(Y_test, Y_pred, output_dict=True)

        # Gets averages
        averages = {idx: report for idx, report in full_report.items() if "avg" in idx}
        # Gets class reports and converts index to class names for readability
        full_report = {
            label_binarizer.classes_[int(idx)]: report
            for idx, report in full_report.items()
            if "avg" not in idx
        }

        # Put the averages back
        full_report = {**averages, **full_report}

        result = {
            "threshold": f"{th:.2f}",
            "precision": f"{p:.2f}",
            "recall": f"{r:.2f}",
            "f1": f"{f1:.2f}",
        }
        results.append(result)

        row_data = (
            result["threshold"],
            result["precision"],
            result["recall"],
            result["f1"],
        )
        print(row(row_data, widths=widths))

    if results_path:
        with open(results_path, "w") as f:
            f.write(json.dumps(results, indent=4))
    if full_report_path:
        with open(full_report_path, "w") as f:
            f.write(json.dumps(full_report, indent=4))


evaluate_model_app = typer.Typer()


@evaluate_model_app.command()
def evaluate_model_cli(
    approach: str = typer.Argument(..., help="model approach e.g.mesh-cnn"),
    model_path: str = typer.Argument(
        ..., help="comma separated paths to pretrained models"
    ),
    data_path: Path = typer.Argument(
        ..., help="path to data that was used for training"
    ),
    label_binarizer_path: Path = typer.Argument(..., help="path to label binarize"),
    threshold: Optional[str] = typer.Option(
        "0.5", help="threshold or comma separated thresholds used to assign tags"
    ),
    results_path: Optional[str] = typer.Option(None, help="path to save results"),
    full_report_path: Optional[str] = typer.Option(
        None,
        help="Path to save full report, i.e. "
        "more comprehensive results than the ones saved in results_path",
    ),
    split_data: bool = typer.Option(
        True, help="flag on whether to split data in same way as was done in train"
    ),
    parameters: str = typer.Option(
        None, help="stringified parameters for model evaluation, if any"
    ),
    config: Optional[Path] = typer.Option(
        None, help="path to config file that defines arguments"
    ),
):
    if config:
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)
        if "ensemble" in cfg:
            approach = cfg["ensemble"]["approach"]
            model_path = cfg["ensemble"]["models"]
            data_path = cfg["ensemble"]["data"]
            label_binarizer_path = cfg["ensemble"]["label_binarizer"]
            threshold = cfg["ensemble"]["threshold"]
            split_data = cfg["ensemble"]["split_data"]  # needs convert to bool
            results_path = cfg["ensemble"].get("results_path", "results.json")
        else:  # I only need to get model parameters which are necessary for loading/predicting
            parameters = cfg["model"]["parameters"]

    if "," in threshold:
        threshold = [float(t) for t in threshold.split(",")]
    else:
        threshold = float(threshold)

    evaluate_model(
        approach,
        model_path,
        data_path,
        label_binarizer_path,
        threshold,
        split_data,
        results_path=results_path,
        full_report_path=full_report_path,
        parameters=parameters,
    )


if __name__ == "__main__":
    evaluate_model_app()
