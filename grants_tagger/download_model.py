import tarfile
import os

from tqdm import tqdm
import requests

MODELS = {
    "disease_mesh": {
        "url": "https://github.com/wellcometrust/grants_tagger/releases/download/v0.1.3/disease_mesh_cnn-2021.03.1.tar.gz",
        "path": "disease_mesh_cnn-2021.03.1.tar.gz",
    }
}


def download_tar(url, path):
    response = requests.get(url, stream=True)
    total_length = int(response.headers.get("content-length"))

    # iif response.status_code == 200:
    with open(path, "wb") as f:
        for chunk in tqdm(
            response.iter_content(chunk_size=1024),
            total=int(total_length / 1024),
            unit="KB",
        ):
            f.write(chunk)


def untar(tar_path, path):
    with tarfile.open(tar_path) as tar:
        tar.extractall(path)


def download_model(model_name):
    if model_name in MODELS:
        url = MODELS[model_name]["url"]
        path = MODELS[model_name]["path"]
        models_path = os.path.join(os.path.dirname(__file__), "../")

        download_tar(url, path)
        untar(path, models_path)
    else:
        print(
            f"{model_name} not recognised. See README.md for list of models available"
        )


if __name__ == "__main__":
    download_model("disease_mesh")
