# Import minimal set of libraries
import configparser
from typing import List, Dict, Union, Any, Optional

from grants_tagger.models.mesh_xlinear import MeshXLinear


def load_model():
    model = MeshXLinear()


def predict_tags(
    X: Union[List, str],
    model_path: str,
    label_binarizer_path: str,
    approach: str = "mesh-xlinear",
    probabilities: bool = False,
    threshold: float = 0.5,
    parameters: Optional[Dict[str, Any]] = None,
    config=None,
):
    """

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

    """
    if config:
        # For some models, it might be necessary to see the parameters before loading it
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)
        parameters = cfg["model"]["parameters"]
