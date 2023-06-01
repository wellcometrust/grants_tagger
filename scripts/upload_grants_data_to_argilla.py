import argilla as rg
import os
import argparse
import json


def upload_data_to_argilla(path: str):
    """
    Uploads data to Argilla. The data should be in jsonl format, with each line containing a json object with the
    following fields:
        - abstract (str): The abstract of the grant.
        - mesh_terms (list): A list of MeSH terms associated with the grant.
    To generate this data, use the create_grants_sample.py script.

    Args:
        path (str): The path to the jsonl file.

    Returns:
        None

    Environment Variables:
        ARGILLA_API_URL (str): The URL of the Argilla API. Defaults to "https://pro.argilla.io".
        ARGILLA_API_KEY (str): The API key for the Argilla API.
    """
    api_url = os.environ.get("ARGILLA_API_URL", "https://pro.argilla.io")
    api_key = os.environ.get("ARGILLA_API_KEY")
    rg.init(
        api_url=api_url,
        api_key=api_key,
        workspace="wellcome",
    )

    with open(path, "r") as f:
        for line in f:
            sample = json.loads(line)
            predictions = sample["mesh_terms"]
            record = rg.TextClassificationRecord(
                text=sample["abstract"],
                prediction=[(term, 1.0) for term in predictions],
                multi_label=True,
            )

            rg.log(record, "grants-test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the jsonl file")
    args = parser.parse_args()

    upload_data_to_argilla(args.path)
