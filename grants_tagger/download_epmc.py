from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm
import requests
import json
import os
import typer


RETRY_PARAMETERS = {
    "backoff_factor": 1.2,
    "status_forcelist": [429, 500, 502, 503, 504],
}

RETRY_BACKOFF_MAX = 310
EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _get_response(session, params):
    response = session.get(EPMC_SEARCH_URL, params=params)
    response_json = response.json()
    return response_json


def yield_results(session, params):
    response_json = _get_response(session, params)

    results = response_json["resultList"]["result"]
    while results:
        for result in results:
            yield result

        params["cursorMark"] = response_json.get("nextCursorMark")
        response_json = _get_response(session, params)

        results = response_json["resultList"]["result"]


def get_hit_count(session, params):
    response_json = _get_response(session, params)
    return response_json["hitCount"]


def download_epmc(download_path, year=2020):
    session = requests.Session()
    retries = Retry(**RETRY_PARAMETERS)
    retries.BACKOFF_MAX = RETRY_BACKOFF_MAX
    session.mount("https://", HTTPAdapter(max_retries=retries))

    year_path = os.path.join(download_path, str(year))
    os.makedirs(year_path, exist_ok=True)
    for month in range(12):
        month = f"{month+1:02}"
        month_path = os.path.join(year_path, f"{month}.jsonl")
        if os.path.exists(month_path):
            print(
                f"Skipping because {month_path} exists. Delete if you want to redownload"
            )
            continue

        tmp_month_path = f"{month_path}.tmp"
        with open(tmp_month_path, "w") as f:
            params = {
                "query": f"(FIRST_PDATE:[{year}-{month}-01 TO {year}-{month}-31])",
                "format": "json",
                "resultType": "core",
                "pageSize": 100,
            }
            hit_count = get_hit_count(session, params)
            for result in tqdm(
                yield_results(session, params),
                total=hit_count,
                desc=f"Year {year} Month {month}",
            ):
                f.write(json.dumps(result))
                f.write("\n")
        os.rename(tmp_month_path, month_path)


download_epmc_app = typer.Typer()


@download_epmc_app.command()
def download_epmc_cli(
    download_path: str = typer.Argument(
        ..., help="path to directory where to download EPMC data"
    ),
    year: int = typer.Option(2020, help="year to download epmc publications"),
):

    download_epmc(download_path, year)


if __name__ == "__main__":
    download_epmc_app()
