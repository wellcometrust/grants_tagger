from scipy import sparse as sp
import numpy as np


def get_params_for_component(params, component):
    """
    Returns a dictionary of all params for one component defined
    in params in the form component__param: value

    e.g.
    >> params = {"vec__min_df": 1, "clf__probability": True}
    >> get_params_for_component(params, "vec")
    {"min_df": 1}
    """
    component_params = {}
    for k, v in params.items():
        if k.startswith(component):
            _, component_arg = k.split(f"{component}__")
            component_params[component_arg] = v
    return component_params


def format_predictions(
    Y_pred_proba, label_binarizer, threshold=0.5, probabilities=True
):
    """
    Formats predictions to output a list of dictionaries

    Y_pred_proba: sparse array or list of predicted probabilites or class
    (i.e. the output of  `.predict` or `.predict_proba` classifier)
    label_binarizer: A sklearn fitted label binarizer
    threshold: Float between 0 and 1
    probabilities: Whether Y_pred_proba will contain probabilities or just predictions

    Returns:
        A list of dictionaries for each prediction, e.g.
        [{"tag": "Malaria", "Probability": 0.5}, ...]
    """
    tags = []
    for y_pred_proba in Y_pred_proba:
        if sp.issparse(y_pred_proba):
            y_pred_proba = np.asarray(y_pred_proba.todense()).ravel()
        if probabilities:
            tags_i = {
                tag: prob
                for tag, prob in zip(label_binarizer.classes_, y_pred_proba)
                if prob >= threshold
            }
        else:
            tags_i = [
                tag
                for tag, prob in zip(label_binarizer.classes_, y_pred_proba)
                if prob >= threshold
            ]
        tags.append(tags_i)

    return tags
