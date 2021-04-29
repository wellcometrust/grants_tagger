from typing import List, Optional
from pathlib import Path
import configparser
import logging
import yaml
import os

import typer

from grants_tagger.evaluate_mti import evaluate_mti
from grants_tagger.evaluate_human import evaluate_human
from grants_tagger.tag_grants import tag_grants
from grants_tagger.pretrain import pretrain as pretrain_model
from grants_tagger.evaluate_model import evaluate_model
from grants_tagger.predict import predict_tags
from grants_tagger.preprocess import preprocess
from grants_tagger.preprocess_mesh import preprocess_mesh
from grants_tagger.train import train_and_evaluate
from grants_tagger.tune_threshold import tune_threshold
from grants_tagger.optimise_params import optimise_params

app = typer.Typer(add_completion=False)

logger = logging.getLogger(__name__)


def convert_dvc_to_sklearn_params(parameters):
    """converts dvc key value params to sklearn nested params if needed"""
    # converts None to empty dict
    if not parameters:
        return {}

    # indication of sklearn pipeline
    has_nested_params = any([v for v in parameters.values() if type(v) is dict])
    if has_nested_params:
        return {
            f"{pipeline_name}__{param_name}": param_value
            for pipeline_name, params in parameters.items()
            for param_name, param_value in params.items()
        }
    else:
        return parameters

@app.command()
def train(
        data_path: Optional[Path] = typer.Argument(None, help="path to processed JSON data to be used for training"),
        label_binarizer_path: Optional[Path] = typer.Argument(None, help="path to label binarizer"),
        model_path: Optional[Path] = typer.Argument(None, help="path to output model.pkl or dir to save model"),
        approach: str = typer.Option("tfidf-svm", help="tfidf-svm, scibert, cnn, ..."),
        parameters: str = typer.Option(None, help="model params in sklearn format e.g. {'svm__kernel: linear'}"),
        test_data_path: Path = typer.Option(None, help="path to processed JSON test data"),
        threshold: float = typer.Option(None, help="threshold to assign a tag"),
        data_format: str = typer.Option("list", help="format that will be used when loading the data. One of list,generator"),
        test_size: float = typer.Option(0.25, help="float or int indicating either percentage or absolute number of test examples"),
        sparse_labels: bool = typer.Option(False, help="flat about whether labels should be sparse when binarized"),
        cache_path: Optional[Path] = typer.Option(None, help="path to cache data transformartions"),
        config: Path = None):

    params_path = os.path.join(os.path.dirname(__file__), "../params.yaml")
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # If parameters not provided from user we initialise from DVC
    if not parameters:
        parameters = params["train"].get(approach)
        parameters = convert_dvc_to_sklearn_params(parameters)
        parameters = str(parameters)
        logger.debug(parameters)

    # Note that config overwrites parameters for backwards compatibility
    if config:
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)

        data_path = cfg["data"]["train_data_path"]
        label_binarizer_path = cfg["model"]["label_binarizer_path"]
        approach = cfg["model"]["approach"]
        parameters = cfg["model"]["parameters"]
        model_path = cfg["model"].get("model_path", None)
        test_data_path = cfg["data"]["test_data_path"]
        threshold = cfg["model"].get("threshold", None)
        if threshold:
            threshold = float(threshold)
        data_format = cfg["data"].get("data_format", "list")
        test_size = float(cfg["data"].get("test_size", 0.25))
        sparse_labels = cfg["model"].get("sparse_labels", False)
        if sparse_labels:
            sparse_labels = bool(sparse_labels)
        cache_path = cfg["data"].get("cache_path")
    
    if model_path and os.path.exists(model_path):
        print(f"{model_path} exists. Remove if you want to rerun.")
    else:

        train_and_evaluate(
            data_path, label_binarizer_path, approach,
            parameters, model_path=model_path,
            test_data_path=test_data_path,
            threshold=threshold,
            data_format=data_format, test_size=test_size,
            sparse_labels=sparse_labels, cache_path=cache_path)


preprocess_app = typer.Typer()


@preprocess_app.command()
def bioasq_mesh(
        input_path: Optional[str] = typer.Argument(None, help="path to BioASQ JSON data"),
        output_path: Optional[str] = typer.Argument(None, help="path to output JSONL data"),
        mesh_metadata_path: Optional[str] = typer.Option(None, help="path to xml file containing MeSH taxonomy"),
        filter_tags: Optional[str] = typer.Option(None, help="filter mesh subbranch like disease"),
        test_split: Optional[float] = typer.Option(None, help="split percentage for test data. if None no split."),
        config: Optional[Path] = typer.Option(None, help="path to config files that defines arguments")):

    params_path = os.path.join(os.path.dirname(__file__), "../params.yaml")
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # Default values from params
    if not filter_tags:
        filter_tags = params["preprocess_bioasq_mesh"]["filter_tags"]

    if config:
        cfg = configparser.ConfigParser()
        cfg.read(config)

        input_path = cfg["preprocess"]["input"]
        output_path = cfg["preprocess"]["output"]
        mesh_metadata_path = cfg["filter_disease_codes"]["mesh_descriptions_file"]
        filter_tags = cfg["filter_disease_codes"].get("filter_tags")
        test_split = cfg["preprocess"].getfloat("test_split")

    # TODO: Refactor with preprocess_mesh
    if test_split:
        data_dir, data_name = os.path.split(output_path)
        train_output_path = os.path.join(data_dir, "train_" + data_name) 
        test_output_path = os.path.join(data_dir, "test_" + data_name)
        
        if os.path.exists(train_output_path) and os.path.exists(test_output_path):
            print(f"{train_output_path} and {test_output_path} exists. Remove them if you want to rerun.")
            return
    else:
        if os.path.exists(output_path):
            print(f"{output_path} exists. Remove if you want to rerun.")
            return

    preprocess_mesh(input_path, output_path, mesh_metadata_path=mesh_metadata_path,
        filter_tags=filter_tags, test_split=test_split)


@preprocess_app.command()
def wellcome_science(
        input_path: Optional[Path] = typer.Argument(None, help="path to raw Excel file with tagged or untagged grant data"),
        output_path: Optional[Path] = typer.Argument(None, help="path to JSONL output file that will be generated"),
        text_cols: Optional[str] = typer.Option(None, help="comma delimited column names to concatenate to text"),
        meta_cols: Optional[str] = typer.Option(None, help="comma delimited column names to include in the meta"),
        config: Path = typer.Option(None, help="path to config file that defines the arguments")):

    params_path = os.path.join(os.path.dirname(__file__), "../params.yaml")
    with open(params_path) as f:
        params = yaml.safe_load(f)

    # Default values from params
    if not text_cols:
        text_cols = params["preprocess_wellcome_science"]["text_cols"]
    if not meta_cols:
        meta_cols = params["preprocess_wellcome_science"]["meta_cols"]

    # Note that config overides values if provided, this ensures backwards compatibility
    if config:
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)

        input_path = cfg["preprocess"]["input"]
        output_path = cfg["preprocess"]["output"]
        text_cols = cfg["preprocess"]["text_cols"]
        if not text_cols:
            text_cols = "Title,Synopsis"
        meta_cols = cfg["preprocess"].get("meta_cols", "Grant_ID,Title")

    text_cols = text_cols.split(",")
    meta_cols = meta_cols.split(",")
    if os.path.exists(output_path):
        print(f"{output_path} exists. Remove if you want to rerun.")
    else:
        preprocess(input_path, output_path, text_cols, meta_cols)


app.add_typer(preprocess_app, name="preprocess")


@app.command()
def predict(
        text: str,
        model_path: Path,
        label_binarizer_path: Path,
        approach: str, 
        probabilities: Optional[bool] = typer.Option(False),
        threshold: Optional[float] = typer.Option(0.5)):
    tags = predict_tags([text], model_path, label_binarizer_path,
                 approach, probabilities, threshold)
    print(tags[0])

evaluate_app = typer.Typer()


@evaluate_app.command()
def model(
        approach: str = typer.Argument(..., help="model approach e.g.mesh-cnn"),
        model_path: str = typer.Argument(..., help="comma separated paths to pretrained models"),
        data_path: Path = typer.Argument(..., help="path to data that was used for training"),
        label_binarizer_path: Path = typer.Argument(..., help="path to label binarize"),
        threshold: Optional[str] = typer.Option("0.5", help="threshold or comma separated thresholds used to assign tags"),
        results_path: str = typer.Option("results.json", help="path to save results"),
        config: Optional[Path] = typer.Option(None, help="path to config file that defines arguments")):

    if config:
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)

        approach = cfg["ensemble"]["approach"]
        model_path = cfg["ensemble"]["models"]
        data_path = cfg["ensemble"]["data"]
        label_binarizer_path = cfg["ensemble"]["label_binarizer"]
        threshold = cfg["ensemble"]["threshold"]
        results_path = cfg["ensemble"].get("results_path", "results.json")

    if "," in threshold:
        threshold = [float(t) for t in threshold.split(",")]
    else:
        threshold = float(threshold)
    evaluate_model(approach, model_path, data_path, label_binarizer_path, threshold, results_path=results_path)


@evaluate_app.command()
def human(data_path: Path, label_binarizer_path: Path):
    evaluate_human(data_path, label_binarizer_path)


@evaluate_app.command()
def mti(
        data_path: Path = typer.Argument(..., help="path to sample JSONL mesh data"),
        label_binarizer_path: Path = typer.Argument(..., help="path to pickled mesh label binarizer"),
        mti_output_path: Path = typer.Argument(..., help="path to mti output txt")):

    evaluate_mti(label_binarizer_path, data_path, mti_output_path)


@evaluate_app.command()
def scispacy(
        data_path: Path = typer.Argument(..., help="JSONL of mesh data that contains text, tags and meta per line"),
        label_binarizer_path: Path = typer.Argument(..., help="label binarizer that transforms mesh names to binarized format"),
        mesh_metadata_path: Path = typer.Argument(..., help="csv that contains metadata about mesh such as UI, Name etc")):

    try:
        from grants_tagger.evaluate_scispacy_meshtagger import evaluate_scispacy_meshtagger
    except ModuleNotFoundError:
        print("Scispacy not installed. To use it install separately pip install -r requirements_scispacy.txt")
    finally:
        evaluate_scispacy_meshtagger(label_binarizer_path, mesh_metatadata_path, data_path)


app.add_typer(evaluate_app, name="evaluate")


@app.command()
def pretrain(
        data_path: Optional[Path] = typer.Argument(None, help="data to pretrain model on"),
        model_path: Optional[Path] = typer.Argument(None, help="path to save mode"),
        model_name: Optional[str] = typer.Option("doc2vec", help="name of model to pretrain"),
        config: Optional[Path] = typer.Option(None, help="config file with arguments for pretrain")):

    if config:
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(config)

        try:
            cfg_pretrain = cfg["pretrain"]
        except KeyError:
            cfg_pretrain = {}
        data_path = cfg_pretrain.get("data_path")
        model_path = cfg_pretrain.get("model_path")
        model_name = cfg_pretrain.get("model_name")

    if not model_path:
        print("No pretraining defined. Skipping.")
    elif os.path.exists(model_path):
        print(f"{model_path} exists. Remove if you want to rerun.")
    else:
        pretrain_model(data_path, model_path, model_name)


tune_app = typer.Typer()


@tune_app.command()
def threshold(
        approach: str = typer.Argument(..., help="modelling approach e.g. mesh-cnn"),
        data_path: Path = typer.Argument(..., help="path to data in jsonl to train and test model"),
        model_path: Path = typer.Argument(..., help="path to data in jsonl to train and test model"),
        label_binarizer_path: Path = typer.Argument(..., help="path to label binarizer"),
        thresholds_path: Path = typer.Argument(..., help="path to save threshold values"),
        sample_size: Optional[int] = typer.Option(None, help="sample size of text data to use for tuning"),
        nb_thresholds: Optional[int] = typer.Option(None, help="number of thresholds to be tried divided evenly between 0 and 1"),
        init_threshold: Optional[float] = typer.Option(None, help="value to initialise threshold values")):

    tune_threshold(
        approach, data_path, model_path, label_binarizer_path,
        thresholds_path, sample_size, nb_thresholds, init_threshold)


@tune_app.command()
def params(
        data_path: Path = typer.Argument(..., help=""),
        label_binarizer_path: Path = typer.Argument(..., help=""),
        approach: str = typer.Argument(..., help=""),
        params: Optional[str] = typer.Option(None, help="")):
    optimise_params(data_path, label_binarizer_path, approach, params=None)


app.add_typer(tune_app, name="tune")


@app.command()
def tag(
        grants_path: Path = typer.Argument(..., help="path to grants csv"),
        tagged_grants_path: Path = typer.Argument(..., help="path to output csv"),
        model_path: Path = typer.Argument(..., help="path to model"),
        label_binarizer_path: Path = typer.Argument(..., help="label binarizer for Y"),
        threshold: float = typer.Option(0.5, help="threshold upon which to assign tag")):

    tag_grants(grants_path, tagged_grants_path, model_path,
               label_binarizer_path, threshold)


@app.command()
def download():
    # download models and data from public s3
    pass


@app.command()
def explain():
    # feature importance for models
    pass


if __name__ == "__main__":
    app(prog_name="grants_tagger")
