import requests
import typer
import json
import os


def submit_test_results(
    test_number,
    tagged_data_path,
    system,
    username=os.environ["BIOASQ_USERNAME"],
    password=os.environ["BIOASQ_PASSWORD"],
):
    with open(tagged_data_path) as f:
        tagged_data = json.load(f)

    url = f"http://participants-area.bioasq.org/tests/uploadResults/{test_number}/"

    data = {
        "username": username,
        "password": password,
        "documents": tagged_data["documents"],
        "system": system,
    }
    response = requests.post(url, data=json.dumps(data))
    print(f"Response code {response.status_code}")
    print(f"Platform answer {response.text}")


if __name__ == "__main__":
    typer.run(submit_test_results)
