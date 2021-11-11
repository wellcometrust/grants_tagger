import tarfile
import logging
import json
import os

from sagemaker.estimator import Estimator
import sagemaker


logger = logging.getLogger(__name__)

ENTRYPOINT = "grants_tagger/train.py"
DEPENDENCIES = [
    "grants_tagger/__init__.py",
    "grants_tagger/utils.py",
    "grants_tagger/models.py",
]
try:
    PROJECTS_BUCKET = os.environ["PROJECTS_BUCKET"]
    PROJECT_NAME = os.environ["PROJECT_NAME"]
    SAGEMAKER_ROLE = os.environ["SAGEMAKER_ROLE"]
except KeyError:
    logger.warning(
        "One or more of PROJECTS_BUCKET, PROJECT_NAME, SAGEMAKER_ROLE is not set. SageMaker will not work."
    )


def upload_code(entrypoint, dependencies, bucket, prefix):
    """
    Upload code for train

    Args:
    entrypoint: path to train script
    dependencies: list of path to additional source files needed
    bucket: s3 bucket that sagemaker will use to read and write artifacts
    prefix: s3 prefix     -//- (same as above)
    """
    session = sagemaker.Session()

    source_files = [entrypoint] + dependencies
    logger.info(source_files)

    tmp_dir = "/tmp"
    filename = os.path.join(tmp_dir, "sourcedir.tar.gz")

    with tarfile.open(filename, mode="w:gz") as t:
        for sf in source_files:
            t.add(sf)

    # Probably can be replaced by boto
    s3_path = session.upload_data(filename, bucket, prefix + "/code")
    return s3_path


def create_hyperparameters(
    entrypoint, code_path, model_path, data_path, label_binarizer_path, **params
):
    """
    Create hyperparameters parameter for sagemaker which is the equivalent of keyword arguments passed in
    CLI or params.yaml etc.

    Args:
    entrypoint: path to train script which will be invoked by Sagemaker
    code_path: path to addtitional source files needed for train
    model_path: path to save model
    data_path: path to data
    label_binarizer_path: path to save label_binarizer
    params: additional params that are passed to train

    Note that paths need to be converted to container paths. Sagemaker is responsible for moving
    data and models in an out of the container.
    Also params are equivalent to the rest of the parameters that are passed through the CLI to
    train. Sagemaker stores all those into hyperapameters and converts to CLI args upon invokation

    Returns:
    hyperaparameters
    """
    # Step 1 - Convert data, model paths
    _, model_subpath = (
        os.path.split(model_path) if model_path else ("", "")
    )  # can be dir or model_name
    _, data_filename = os.path.split(data_path)
    _, label_binarizer_filename = os.path.split(label_binarizer_path)

    container_model_path = f"/opt/ml/model/{model_subpath}" if model_subpath else ""
    container_data_path = f"/opt/ml/input/data/training/{data_filename}"
    container_label_binarizer_path = f"/opt/ml/model/{label_binarizer_filename}"

    hyperparameters = {
        "data_path": container_data_path,
        "model_path": container_model_path,
        "label_binarizer_path": container_label_binarizer_path,
    }

    # Step 2 - Add additional params passed to train in hyperparams
    for k, v in params.items():
        hyperparameters[k] = v

    # Step 3 - Define entrypoint to invoke and additional source files needed
    hyperparameters["sagemaker_program"] = entrypoint
    hyperparameters["sagemaker_submit_directory"] = code_path

    # Step 4 - Package hyperparams params in json
    hyperparameters = {str(k): json.dumps(v) for (k, v) in hyperparameters.items() if v}
    return hyperparameters


def train_with_sagemaker(instance_type="local", config_version=None, **kwargs):
    """Invokes train with kwargs using Sagemaker using instance type"""
    code_path = upload_code(
        ENTRYPOINT, DEPENDENCIES, bucket=PROJECTS_BUCKET, prefix=PROJECT_NAME
    )
    logger.info(code_path)

    hyperparameters = create_hyperparameters(ENTRYPOINT, code_path, **kwargs)
    logger.info(hyperparameters)

    data_path = kwargs["data_path"]
    if instance_type == "local":
        model_path = "file://"
        data_path = f"file://{data_path}"
    else:
        model_path = f"s3://{PROJECTS_BUCKET}/{PROJECT_NAME}/output/"
        data_path = f"s3://{PROJECTS_BUCKET}/{PROJECT_NAME}/{data_path}"
    logger.info(f"model_path: {model_path}")
    logger.info(f"data_path: {data_path}")

    base_job_name = kwargs["approach"]
    if config_version:
        config_version = config_version.replace(".", "-")
        base_job_name = f"{base_job_name}-{config_version}"
    es = Estimator(
        image_uri=os.environ["ECR_IMAGE"],
        role=SAGEMAKER_ROLE,
        framework_version="0.20.0",
        instance_count=1,
        instance_type=instance_type,  # "local", #,"ml.m5.large",
        output_path=model_path,
        hyperparameters=hyperparameters,
        base_job_name=base_job_name,
        max_run=5 * 24 * 60 * 60,  # 5 days
    )
    es.fit({"training": data_path})
