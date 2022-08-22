# Import minimal set of libraries
import ast
import configparser
import pickle
from typing import List, Dict, Union, Any, Optional

from grants_tagger.models.mesh_xlinear import MeshXLinear
from grants_tagger.models.utils import format_predictions


def predict_tags(
    X: Union[List, str],
    model_path: str,
    label_binarizer_path: str,
    probabilities: bool = False,
    threshold: float = 0.5,
    parameters: Optional[Dict[str, Any]] = None,
    config=None,
) -> List[List[Dict]]:
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
        A list of lists of dictionaries for each prediction, e.g.
        [[{"tag": "Malaria", "Probability": 0.5}, ...], [...]]

    """
    # Reads from config file (if needs be)

    if config:
        # For some models, it might be necessary to see the parameters before loading it
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)
        parameters = cfg["model"]["parameters"]

    # Loads model and sets parameters appropriately
    model = MeshXLinear()

    parameters = ast.literal_eval(parameters)
    model.set_params(**parameters)
    model.load(model_path)

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.loads(f.read())

    Y_pred_proba = model.predict_proba(X)
    tags = format_predictions(
        Y_pred_proba, label_binarizer, threshold=threshold, probabilities=probabilities
    )

    return tags
