from unittest import mock
import json
import os

import pytest

from grants_tagger.download_epmc import download_epmc


@pytest.fixture
def download_path(tmp_path):
    return tmp_path


def create_month_data(month_path):
    with open(month_path, "w") as f:
        f.write(json.dumps({"item": "fake"}))
        f.write("\n")


@pytest.mark.inference_time
@mock.patch("grants_tagger.download_epmc.get_hit_count", return_value=5)
@mock.patch("grants_tagger.download_epmc.yield_results", return_value=["item"])
def test_download_epmc(mock_get_hit_count, mock_yield_results, download_path):
    year = 2020
    download_epmc(download_path, year)
    for month in range(12):
        month_path = os.path.join(download_path, str(year), f"{month+1:02}.jsonl")
        assert os.path.exists(month_path)


@pytest.mark.inference_time
@mock.patch("grants_tagger.download_epmc.get_hit_count", return_value=5)
@mock.patch("grants_tagger.download_epmc.yield_results", return_value=["item"])
def test_download_epmc_skip(mock_get_hit_count, mock_yield_results, download_path):
    year = 2020
    year_path = os.path.join(download_path, str(year))
    os.makedirs(year_path)
    month_path = os.path.join(year_path, f"01.jsonl")
    create_month_data(month_path)
    download_epmc(download_path, year)
    with open(month_path) as f:
        for line in f:
            item = json.loads(line)
            break
    assert item["item"] == "fake"


@pytest.mark.inference_time
@mock.patch("grants_tagger.download_epmc.get_hit_count", return_value=5)
@mock.patch(
    "grants_tagger.download_epmc.yield_results", return_value=[{"item": "not fake"}]
)
def test_download_redownload_tmp(mock_get_hit_count, mock_yield_results, download_path):
    year = 2020
    year_path = os.path.join(download_path, str(year))
    os.makedirs(year_path)
    month_path = os.path.join(year_path, f"01.jsonl")
    tmp_month_path = f"{month_path}.tmp"
    create_month_data(tmp_month_path)
    download_epmc(download_path, year)
    with open(month_path) as f:
        for line in f:
            item = json.loads(line)
            break
    assert item["item"] == "not fake"
