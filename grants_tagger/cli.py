from typing import Optional
from pathlib import Path
import subprocess
import logging
import os

import typer


logger = logging.getLogger(__name__)


from grants_tagger.train import train_cli

try:
    from grants_tagger.train_with_sagemaker import train_with_sagemaker_cli
except ModuleNotFoundError as e:
    logger.warning("Sagemaker missing so training with sagemaker not working.")
    logger.debug(e)

from grants_tagger.preprocess_mesh import preprocess_mesh_app
from grants_tagger.preprocess_wellcome import preprocess_wellcome_app
from grants_tagger.predict import predict_app
from grants_tagger.evaluate_human import evaluate_human_app
from grants_tagger.evaluate_mesh_on_grants import evaluate_mesh_on_grants_app
from grants_tagger.evaluate_model import evaluate_model_app
from grants_tagger.evaluate_mti import evaluate_mti_app
from grants_tagger.evaluate_scispacy_meshtagger import evaluate_scispacy_meshtagger_app
from grants_tagger.pretrain import pretrain_app
from grants_tagger.tune_threshold import tune_threshold_app
from grants_tagger.optimise_params import tune_params_app
from grants_tagger.download_epmc import download_epmc_app
from grants_tagger.download_model import download_model_app
from grants_tagger.explain import explain_app

app = typer.Typer(add_completion=False)


# Move to train and import from there
@app.command()
def train(
    data_path: Optional[Path] = typer.Argument(
        None, help="path to processed JSON data to be used for training"
    ),
    label_binarizer_path: Optional[Path] = typer.Argument(
        None, help="path to label binarizer"
    ),
    model_path: Optional[Path] = typer.Argument(
        None, help="path to output model.pkl or dir to save model"
    ),
    approach: str = typer.Option("tfidf-svm", help="tfidf-svm, scibert, cnn, ..."),
    parameters: str = typer.Option(
        None, help="model params in sklearn format e.g. {'svm__kernel: linear'}"
    ),
    threshold: float = typer.Option(None, help="threshold to assign a tag"),
    data_format: str = typer.Option(
        "list",
        help="format that will be used when loading the data. One of list,generator",
    ),
    train_info: str = typer.Option(None, help="path to train times and instance"),
    sparse_labels: bool = typer.Option(
        False, help="flat about whether labels should be sparse when binarized"
    ),
    cache_path: Optional[Path] = typer.Option(
        None, help="path to cache data transformartions"
    ),
    config: Path = None,
    cloud: bool = typer.Option(False, help="flag to train using Sagemaker"),
    instance_type: str = typer.Option(
        "local", help="instance type to use when training with Sagemaker"
    ),
):

    if cloud:
        train_with_sagemaker_cli(
            data_path=data_path,
            label_binarizer_path=label_binarizer_path,
            model_path=model_path,
            approach=approach,
            parameters=parameters,
            threshold=threshold,
            data_format=data_format,
            train_info=train_info,
            sparse_labels=sparse_labels,
            cache_path=cache_path,
            config=config,
            instance_type=instance_type,
        )
    else:
        logger.info(parameters)
        train_cli(
            data_path=data_path,
            label_binarizer_path=label_binarizer_path,
            model_path=model_path,
            approach=approach,
            parameters=parameters,
            threshold=threshold,
            data_format=data_format,
            sparse_labels=sparse_labels,
            cache_path=cache_path,
            config=config,
        )


preprocess_app = typer.Typer()
preprocess_app.add_typer(preprocess_mesh_app, name="bioasq-mesh")
preprocess_app.add_typer(preprocess_wellcome_app, name="wellcome-science")
app.add_typer(preprocess_app, name="preprocess")

app.add_typer(predict_app, name="predict")

evaluate_app = typer.Typer()
evaluate_app.add_typer(evaluate_mesh_on_grants_app, name="mesh")
evaluate_app.add_typer(evaluate_model_app, name="model")
evaluate_app.add_typer(evaluate_mti_app, name="mti")
evaluate_app.add_typer(evaluate_human_app, name="human")
evaluate_app.add_typer(evaluate_scispacy_meshtagger_app, name="scispacy")
app.add_typer(preprocess_app, name="evaluate")

app.add_typer(pretrain_app, name="pretrain")

tune_app = typer.Typer()
tune_app.add_typer(tune_threshold_app, name="threshold")
tune_app.add_typer(tune_params_app, name="params")
app.add_typer(tune_app, name="tune")

download_app = typer.Typer()
download_app.add_typer(download_epmc_app, name="epmc-mesh")
download_app.add_typer(download_model_app, name="model")
app.add_typer(download_app, name="download")

app.add_typer(explain_app, name="explain")


@app.command()
def visualize():
    st_app_path = os.path.join(os.path.dirname(__file__), "streamlit_visualize.py")
    subprocess.Popen(["streamlit", "run", st_app_path])


if __name__ == "__main__":
    app()
