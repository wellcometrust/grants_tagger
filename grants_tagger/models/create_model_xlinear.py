import ast
from grants_tagger.utils import load_pickle
from grants_tagger.models.mesh_xlinear import MeshXLinear


class ApproachNotImplemented(Exception):
    pass


def create_model(parameters=None):
    """Creates an XLinear model

    Args:
    - parameters: parameters to set the XLinear model

    Returns:
        an xlinear model

    """
    model = MeshXLinear()

    if parameters:
        params = ast.literal_eval(parameters)
        model.set_params(**params)
    else:
        parameters = {}
    return model


def load_model(model_path, parameters=None):
    """Loads an XLinear model from a pickle file

    Args:
    - model_path: location of pickled model
    - parameters: parameters to set the XLinear model

    Returns:
        an xlinear model

    """
    if str(model_path).endswith(".pkl"):
        model = load_pickle(model_path)
    else:
        model = create_model(parameters=parameters)
        model.load(model_path)

    return model
