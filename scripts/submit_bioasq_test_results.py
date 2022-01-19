import requests
import typer
import json


def submit_test_results(test_number, tagged_data_path, system, username, password):
    tagged_data = json.load(tagged_data_path)

    url = f"http://participants-area.bioasq.org/tests/uploadResults/{test_number}/"

    data = {
        "username": username,
        "password": password,
        "documents": tagged_data["tagged_data"]
        "system": system
    }
    response = requests.post(url, data=json.dumps(data))
    print(f"Response code {reponse.status_code}")

if __name__ == "__main__":
    typer.run(submit_test_results)
