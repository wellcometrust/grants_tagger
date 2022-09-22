# Import minimal set of libraries
import ast
import configparser
import json
import os
import pickle
from typing import List, Dict, Union, Any, Optional

from sklearn.metrics import precision_recall_fscore_support, classification_report

from grants_tagger.models.mesh_xlinear import MeshXLinear
from grants_tagger.models.utils import format_predictions
from grants_tagger.utils import load_train_test_data
from grants_tagger.label_binarizer import create_label_binarizer


def read_config(config_path):
    # Reads from config file (if needs be)
    # For some models, it might be necessary to see the parameters before loading it
    cfg = configparser.ConfigParser(allow_no_value=True)
    cfg.read(config_path)
    model_dict = cfg["model"]

    return model_dict


def evaluate(
    model,
    label_binarizer,
    train_data_path,
    test_data_path,
    results_path,
    full_report_path,
    threshold=0.5,
):

    _, X_test, _, Y_test = load_train_test_data(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        label_binarizer=label_binarizer,
    )

    print("Evaluating model")
    Y_pred_proba = model.predict_proba(X_test)
    Y_pred = Y_pred_proba > threshold
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
        "threshold": f"{threshold:.2f}",
        "precision": f"{p:.2f}",
        "recall": f"{r:.2f}",
        "f1": f"{f1:.2f}",
    }

    if results_path:
        with open(results_path, "w") as f:
            f.write(json.dumps(result, indent=4))
    if full_report_path:
        with open(full_report_path, "w") as f:
            f.write(json.dumps(full_report, indent=4))

    return result, full_report


def train(
    train_data_path: str,
    label_binarizer_path: str,
    model_path: str,
    parameters: Optional[Dict[str, Any]] = None,
    config: Optional[str] = None,
    sparse_labels: bool = True,
    threshold: float = 0.5,
):
    if config:
        # For some models, it might be necessary to see the parameters before loading it
        config_dict = read_config(config)
        parameters = ast.literal_eval(config_dict["parameters"])
        model_path = model_path or config_dict["model_path"]
        label_binarizer_path = (
            label_binarizer_path or config_dict["label_binarizer_path"]
        )

    if os.path.exists(label_binarizer_path):
        print(f"{label_binarizer_path} exists. Loading existing")
        with open(label_binarizer_path, "rb") as f:
            label_binarizer = pickle.loads(f.read())
    else:
        label_binarizer = create_label_binarizer(
            train_data_path, label_binarizer_path, sparse_labels
        )

    # Reads from config file (if needs be)

    # Loads model and sets parameters appropriately
    model = MeshXLinear()

    # Turns None into {}
    parameters = parameters or {}

    model.set_params(**parameters)
    model.threshold = threshold

    # X can be (numpy arrays, lists) or generators
    X_train, _, Y_train, _ = load_train_test_data(
        train_data_path=train_data_path,
        label_binarizer=label_binarizer,
    )

    print("Fitting model")
    model.fit(X_train, Y_train)

    os.makedirs(model_path, exist_ok=True)

    print(f"Saving model on {model_path}")
    model.save(model_path)

    return model, label_binarizer


def predict_tags(
    X: Union[List, str],
    model_path: str,
    label_binarizer_path: str,
    probabilities: bool = False,
    threshold: float = 0.5,
    parameters: Optional[Dict[str, Any]] = None,
    config: Optional[str] = None,
) -> Union[List[Dict], List[List[Dict]]]:
    """
    Slim function to predict on tags for MeshXLinear, by passing create_model.py (which is a very heavy module)

    Args:
        X: list or numpy array of texts
        model_path: path to trained model
        label_binarizer_path: path to trained label_binarizer
        approach: approach used to train the model
        probabilities: bool, default False. When true probabilities are returned along with tags
        threshold: float, default 0.5. Probability threshold to be used to assign tags.
        parameters: any params required upon model creation
        config: Path to config file

    Returns:
        A list of dictionaries (or list of lists of dictionaries for each prediction), e.g.
        [[{"tag": "Malaria", "Probability": 0.5}, ...], [...]]

    """
    # Converts X to a singleton if X is string
    x_is_string = isinstance(X, str)
    X = [X] if x_is_string else X

    # Reads from config file (if needs be)
    if config:
        # For some models, it might be necessary to see the parameters before loading it
        parameters = read_config(config)

    # Loads model and sets parameters appropriately
    model = MeshXLinear()

    # Turns None into {}
    parameters = parameters or {}

    model.set_params(**parameters)
    model.load(model_path)

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    Y_pred_proba = model.predict_proba(X)
    tags = format_predictions(
        Y_pred_proba, label_binarizer, threshold=threshold, probabilities=probabilities
    )

    if x_is_string:
        tags = tags[0]

    return tags
