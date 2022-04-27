import tempfile
import json
import os

from grants_tagger.preprocess_mesh import process_data, preprocess_mesh


def test_preprocess_mesh():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"],
        "journal": "Journal",
        "year": 2018,
    }
    with tempfile.NamedTemporaryFile(mode="r+") as input_tmp:
        input_tmp.write("\n" + json.dumps(item) + ", ")
        input_tmp.seek(0)
        output_tmp = tempfile.NamedTemporaryFile(mode="r+")
        preprocess_mesh(input_tmp.name, output_tmp.name)
    output_tmp.seek(0)
    expected_processed_item = {
        "text": "This is an abstract",
        "tags": ["T1", "T2"],
        "meta": {"journal": "Journal", "year": 2018},
    }
    assert output_tmp.read() == json.dumps(expected_processed_item) + "\n"


def test_process_data():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"],
        "journal": "Journal",
        "year": 2018,
    }
    expected_processed_item = {
        "text": "This is an abstract",
        "tags": ["T1", "T2"],
        "meta": {"journal": "Journal", "year": 2018},
    }
    processed_item = process_data(item)
    assert processed_item == expected_processed_item


def test_process_data_with_filter_tags():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"],
        "journal": "Journal",
        "year": 2018,
    }
    expected_processed_item = {
        "text": "This is an abstract",
        "tags": ["T1"],
        "meta": {"journal": "Journal", "year": 2018},
    }
    processed_item = process_data(item, filter_tags=["T1"])
    assert processed_item == expected_processed_item


def test_process_data_with_missing_filter_tag():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"],
        "journal": "Journal",
        "year": 2018,
    }
    processed_item = process_data(item, filter_tags=["T3"])
    assert processed_item == None


def test_process_data_with_filter_years():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"],
        "journal": "Journal",
        "year": 2018,
    }
    processed_item = process_data(item, filter_years="2019,2020")
    assert processed_item == None
    item["year"] = 2020
    expected_processed_item = {
        "text": "This is an abstract",
        "tags": ["T1", "T2"],
        "meta": {"journal": "Journal", "year": 2020},
    }
    processed_item = process_data(item, filter_years="2019,2020")
    assert processed_item == expected_processed_item
