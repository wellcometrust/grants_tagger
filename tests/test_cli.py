import tempfile
import os.path
import pickle
import json
import csv

from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from typer.testing import CliRunner
import pytest

from grants_tagger.cli import app
from grants_tagger.utils import convert_dvc_to_sklearn_params
from grants_tagger.models.mesh_xlinear import MeshXLinear
from grants_tagger.models.create_model_transformer import create_model
from grants_tagger.tune_threshold import tune_threshold

runner = CliRunner()


DATA = [
    {
        "text": "this grant is about Malaria",
        "tags": ["Malaria"],
        "meta": {"Tagger1_tags": ["1"]},
    },
    {
        "text": "this one is about Malaria and Cholera",
        "tags": ["Malaria", "Cholera"],
        "meta": {"Tagger1_tags": ["2"]},
    },
    {
        "text": "mainly about Hepatitis",
        "tags": ["Hepatitis"],
        "meta": {"Tagger1_tags": ["3"]},
    },
    {
        "text": "both Cholera and Hepatitis and maybe about Arbovirus too",
        "tags": ["Cholera", "Hepatitis", "Arbovirus"],
        "meta": {"Tagger1_tags": ["4"]},
    },
    {"text": "new one Dengue", "tags": ["Dengue"], "meta": {"Tagger1_tags": ["5"]}},
    {
        "text": "both Dengue Malaria",
        "tags": ["Dengue", "Malaria"],
        "meta": {"Tagger1_tags": ["6"]},
    },
    {
        "text": "this grant is about Arbovirus",
        "tags": ["Arbovirus"],
        "meta": {"Tagger1_tags": ["7"]},
    },
]


BIOASQ_DATA = {
    "articles": [
        {"abstractText": "Malaria", "MeshMajor": ["Malaria"]},
        {"abstractText": "HIV", "MeshMajor": ["HIV"]},
    ]
}


PRETRAIN_DATA = [
    {"synopsis": "Malaria is a disase"},
    {"synopsis": "HIV is another disease"},
]


MTI_DATA = [
    ("0", "T1", "", ""),
    ("1", "T1", "", ""),
    ("2", "T1", "", ""),
    ("3", "T1", "", ""),
    ("4", "T1", "", ""),
    ("5", "T1", "", ""),
    ("6", "T1", "", ""),
]


MESH_TAGS = [
    {"DescriptorUI": "D008288", "DescriptorName": "Malaria"},
    {"DescriptorUI": "D006678", "DescriptorName": "HIV"},
]


MESH_DATA = [
    {
        "text": "Malaria is a mosquito-borne infectious disease that affects humans and other animals. Malaria causes symptoms that typically include fever, tiredness, vomiting, and headaches. In severe cases it can cause yellow skin, seizures, coma, or death",
        "tags": ["Malaria"],
        "meta": {},
    },
    {
        "text": "The human immunodeficiency viruses are two species of Lentivirus that infect humans. Without treatment, average survival time after infection with HIV is estimated to be 9 to 11 years, depending on the HIV subtype.",
        "tags": ["HIV"],
        "meta": {},
    },
]


def write_jsonl(data_path, data):
    with open(data_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def write_csv(data_path, data, delimiter=","):
    fieldnames = list(data[0].keys())
    with open(data_path, "w") as f:
        csvwriter = csv.DictWriter(f, delimiter=delimiter, fieldnames=fieldnames)
        csvwriter.writeheader()
        for line in data:
            csvwriter.writerow(line)


def write_pickle(path, obj):
    with open(path, "wb") as f:
        f.write(pickle.dumps(obj))


def read_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.loads(f.read())
    return obj


def save_transformer_model_to_path(save_path):
    model = create_model()
    model.load("Wellcome/WellcomeBertMesh")
    model.save(save_path)


def create_label_binarizer(label_binarizer_path, data, sparse_labels=False):
    model = create_model()
    model.load("Wellcome/WellcomeBertMesh")

    all_tags = list(model.model.id2label.values())
    label_binarizer = MultiLabelBinarizer(sparse_output=sparse_labels)
    label_binarizer.fit([all_tags])

    write_pickle(label_binarizer_path, label_binarizer)


@pytest.mark.train_command
def test_train_command():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "data.jsonl")
        model_path = os.path.join(tmp_dir, "model.pkl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        train_info_path = os.path.join(tmp_dir, "train_info.json")

        write_jsonl(data_path, DATA)

        result = runner.invoke(
            app,
            [
                "train",
                data_path,
                label_binarizer_path,
                model_path,
                "--parameters",
                "{'vectorizer_library': 'sklearn', 'ngram_range': (1, 1),'beam_size': 30, 'threshold': 0.1, 'only_topk': 200, 'min_df':1, 'min_weight_value': 0.1, 'max_features':200}",
                "--train-info",
                train_info_path,
                "--sparse-labels",
                "--slim",
                "--config",
                "configs/mesh/test_train.ini",
            ],
        )
        assert result.exit_code == 0
        assert os.path.exists(model_path)
        assert os.path.isfile(label_binarizer_path)
        assert os.path.isfile(train_info_path)


def test_preprocess_bioasq_mesh_command():
    # Needs mesh xml as input
    pass


def test_preprocess_wellcome_science_command():
    # Needs raw science data as input
    pass


def test_convert_dvc_to_sklearn_params():
    params = None
    sklearn_params = convert_dvc_to_sklearn_params(params)
    assert sklearn_params == {}

    # no conversion needed
    params = {"learning_rate": 1e-5}
    sklearn_params = convert_dvc_to_sklearn_params(params)
    assert params == sklearn_params

    params = {
        "tfidf": {"ngrams": [1, 2]},
        "svm": {"estimator__class_weight": "balanced"},
    }
    expected_params = {
        "tfidf__ngrams": [1, 2],
        "svm__estimator__class_weight": "balanced",
    }
    sklearn_params = convert_dvc_to_sklearn_params(params)
    assert sklearn_params == expected_params


def test_pretrain_command():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "data.csv")
        model_path = os.path.join(tmp_dir, "model")

        write_csv(data_path, PRETRAIN_DATA)

        result = runner.invoke(app, ["pretrain", data_path, model_path])
        assert result.exit_code == 0
        assert os.path.isdir(model_path)


@pytest.mark.inference_time
def test_predict_command():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir)
        save_transformer_model_to_path(model_path)

        text = "malaria"
        result = runner.invoke(app, ["predict", text, model_path])
        print(result)
        assert result.exit_code == 0


def test_evaluate_model_command():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir)
        data_path = os.path.join(tmp_dir, "data.jsonl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")

        create_label_binarizer(label_binarizer_path, MESH_DATA, sparse_labels=True)
        save_transformer_model_to_path(model_path)
        write_jsonl(data_path, MESH_DATA)

        result = runner.invoke(
            app,
            [
                "evaluate",
                "model",
                model_path,
                data_path,
                label_binarizer_path,
            ],
        )
        assert result.exit_code == 0


def test_evaluate_human_command():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "data.jsonl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")

        create_label_binarizer(label_binarizer_path, DATA)
        write_jsonl(data_path, DATA)

        result = runner.invoke(
            app, ["evaluate", "human", data_path, label_binarizer_path]
        )
        assert result.exit_code == 0


@pytest.mark.skip(reason="import fail when scispacy installed")
@pytest.mark.scispacy
def test_evaluate_scispacy_command():
    with tempfile.TemporaryDirectory() as tmp_dir:
        mesh_data_path = os.path.join(tmp_dir, "mesh_data.jsonl")
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        mesh_metadata_path = os.path.join(tmp_dir, "mesh_metadata.jsonl")

        write_jsonl(mesh_data_path, MESH_DATA)
        write_jsonl(mesh_metadata_path, MESH_TAGS)
        create_label_binarizer(label_binarizer_path, MESH_DATA)

        result = runner.invoke(
            app,
            [
                "evaluate",
                "scispacy",
                mesh_data_path,
                label_binarizer_path,
                mesh_metadata_path,
            ],
        )
        assert result.exit_code == 0


def test_evaluate_mti_command():
    with tempfile.TemporaryDirectory() as tmp_dir:
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        data_path = os.path.join(tmp_dir, "data.jsonl")
        mti_data_path = os.path.join(tmp_dir, "mti.csv")

        with open(mti_data_path, "w") as f:
            for line in MTI_DATA:
                f.write("|".join(line) + "\n")
        write_jsonl(data_path, DATA)
        create_label_binarizer(label_binarizer_path, DATA)

        result = runner.invoke(
            app, ["evaluate", "mti", data_path, label_binarizer_path, mti_data_path]
        )
        assert result.exit_code == 0


def test_tune_threshold_command():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = os.path.join(tmp_dir, "data.jsonl")
        model_path = "Wellcome/WellcomeBertMesh"
        label_binarizer_path = os.path.join(tmp_dir, "label_binarizer.pkl")
        thresholds_path = os.path.join(tmp_dir, "thresholds.pkl")

        write_jsonl(data_path, MESH_DATA)
        create_label_binarizer(label_binarizer_path, MESH_DATA, sparse_labels=True)
        save_transformer_model_to_path(model_path)

        # result = runner.invoke(
        #     app,
        #     [
        #         "tune",
        #         "threshold",
        #         data_path,
        #         model_path,
        #         label_binarizer_path,
        #         thresholds_path,
        #     ],
        # )

        tune_threshold(
            data_path,
            model_path,
            label_binarizer_path,
            thresholds_path,
        )

        # assert result.exit_code == 0
        assert os.path.isfile(thresholds_path)


def test_tune_params_command():
    # no test for optimise params yet
    pass


def test_tag_science_ensemble():
    pass


def test_tag_mesh_cnn():
    pass


def test_tag_zero_threshold():
    pass
