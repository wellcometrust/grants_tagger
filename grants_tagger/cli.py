from typing import Optional
from pathlib import Path
import subprocess
import logging
import os
import time
import json

import typer
import dvc.api

logger = logging.getLogger(__name__)

try:
    from grants_tagger.train import train_cli
except ModuleNotFoundError as e:
    logger.warning(
        "Train_cli couldn't be imported, probably due to a missing dependency."
        " See error below (you can probably still use train_slim)"
    )
    logger.debug(e)

try:
    from grants_tagger.train_with_sagemaker import train_with_sagemaker_cli
except ModuleNotFoundError as e:
    logger.warning("Sagemaker missing so training with sagemaker not working.")
    logger.debug(e)

from grants_tagger.slim import mesh_xlinear

from grants_tagger.preprocess_mesh import preprocess_mesh_cli
from grants_tagger.preprocess_wellcome import preprocess_wellcome_cli
from grants_tagger.predict import predict_cli
from grants_tagger.evaluate_human import evaluate_human_cli
from grants_tagger.evaluate_mesh_on_grants import evaluate_mesh_on_grants_cli
from grants_tagger.evaluate_model import evaluate_model_cli
from grants_tagger.evaluate_mti import evaluate_mti_cli
from grants_tagger.evaluate_scispacy_meshtagger import evaluate_scispacy_cli
from grants_tagger.pretrain import pretrain_cli
from grants_tagger.tune_threshold import tune_threshold_cli
from grants_tagger.optimise_params import tune_params_cli
from grants_tagger.download_epmc import download_epmc_cli
from grants_tagger.download_model import download_model_cli

# from grants_tagger.explain import explain_cli

from grants_tagger.utils import get_ec2_instance_type

app = typer.Typer()


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
    slim: bool = typer.Option(
        False,
        help="flag to tell whether the model is slim (at the moment just xlinear)",
    ),
    cloud: bool = typer.Option(False, help="flag to train using Sagemaker"),
    instance_type: str = typer.Option(
        "local", help="instance type to use when training with Sagemaker"
    ),
):
    start = time.time()
    if slim:
        dvc_params = dvc.api.params_show()

        config = config or dvc_params.get("params.yaml:train", {}).get(
            approach, {}
        ).get("config")

        logging.info(f"Training with config file: {config}")

        mesh_xlinear.train(
            train_data_path=data_path,
            label_binarizer_path=label_binarizer_path,
            model_path=model_path,
            parameters=parameters,
            config=config,
            threshold=threshold,
            sparse_labels=sparse_labels,
        )

    elif cloud:
        train_with_sagemaker_cli(
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
            train_info=train_info,
            sparse_labels=sparse_labels,
            cache_path=cache_path,
            config=config,
        )

    duration = time.time() - start
    instance = get_ec2_instance_type()
    print(f"Took {duration:.2f} to train")
    if train_info:
        with open(train_info, "w") as f:
            json.dump({"duration": duration, "ec2_instance": instance}, f)


preprocess_app = typer.Typer()
preprocess_app.command("bioasq-mesh")(preprocess_mesh_cli)
preprocess_app.command("wellcome-science")(preprocess_wellcome_cli)
app.add_typer(preprocess_app, name="preprocess")

app.command("predict")(predict_cli)

evaluate_app = typer.Typer()
evaluate_app.command("grants")(evaluate_mesh_on_grants_cli)
evaluate_app.command("model")(evaluate_model_cli)
evaluate_app.command("mti")(evaluate_mti_cli)
evaluate_app.command("human")(evaluate_human_cli)
evaluate_app.command("scispacy")(evaluate_scispacy_cli)
app.add_typer(evaluate_app, name="evaluate")

app.command("pretrain")(pretrain_cli)

tune_app = typer.Typer()
tune_app.command("threshold")(tune_threshold_cli)
tune_app.command("params")(tune_params_cli)
app.add_typer(tune_app, name="tune")

download_app = typer.Typer()
download_app.command("epmc-mesh")(download_epmc_cli)
download_app.command("model")(download_model_cli)
app.add_typer(download_app, name="download")

# app.command("explain")(explain_cli)


@app.command()
def visualize():
    st_app_path = os.path.join(os.path.dirname(__file__), "streamlit_visualize.py")
    subprocess.Popen(["streamlit", "run", st_app_path])


if __name__ == "__main__":
    app()
