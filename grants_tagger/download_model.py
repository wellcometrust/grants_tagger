import tarfile
import os
import typer

from tqdm import tqdm
import requests

from grants_tagger import __version__ as version

MODELS = {
    "mesh": {
        "url": f"https://datalabs-public.s3.eu-west-2.amazonaws.com/grants_tagger/models/xlinear-{version}.tar.gz",
        "path": f"xlinear-{version}.tar.gz",
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
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path)


def download_model(model_name):
    if model_name in MODELS:
        url = MODELS[model_name]["url"]
        path = MODELS[model_name]["path"]
        models_path = "."  # model tarball contains models/

        if os.path.exists(path):
            print(f"{path} exists. skipping download.")
        else:
            print(f"download model in {path}")
            download_tar(url, path)
        print("untar model")
        untar(path, models_path)
    else:
        print(
            f"{model_name} not recognised. See README.md for list of models available"
        )


download_model_app = typer.Typer()


@download_model_app.command()
def download_model_cli(
    model_name: str = typer.Argument(
        ..., help="model name to download e.g. disease_mesh"
    )
):
    download_model(model_name)


if __name__ == "__main__":
    download_model_app()
