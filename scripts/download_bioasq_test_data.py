import requests
import typer
import json


def download_test_data(test_number, username, password, data_path):
    url = f"http://participants-area.bioasq.org/tests/{test_number}"

    response = requests.get(url, auth=(username, password))
    data = json.loads(response.text)
    print(f"Test size is {len(data)}")

    with open(data_path, "w") as f:
        f.write(json.dumps(data))

if __name__ == "__main__":
    typer.run(download_test_data)
