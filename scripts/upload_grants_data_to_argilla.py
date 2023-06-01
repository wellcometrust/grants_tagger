import argilla as rg
import os
import argparse
import json


def upload_data_to_argilla(path: str):
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
