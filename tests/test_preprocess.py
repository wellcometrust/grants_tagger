import pandas as pd
import numpy as np

from grants_tagger.preprocess_wellcome import yield_preprocess_data


def test_multiple_tags():
    data = pd.DataFrame(
        {
            "Title": ["A", "A"],
            "Synopsis": ["a", "a"],
            "Lay Summary": ["b", "b"],
            "Qu.": ["c", "c"],
            "Scheme": ["d", "d"],
            "Team": ["e", "e"],
            "Grant_ID": [1, 1],
            "Sciencetags": ["T1", "T2"],
        }
    )
    test_processed_data = [
        {"text": "A a", "meta": {"Grant_ID": 1, "Title": "A"}, "tags": ["T1", "T2"]}
    ]
    out = list(yield_preprocess_data(data))

    assert out == test_processed_data


def test_multiple_grants():
    data = pd.DataFrame(
        {
            "Title": ["A", "B"],
            "Synopsis": ["a", "b"],
            "Lay Summary": ["b", "b"],
            "Qu.": ["c", "c"],
            "Scheme": ["d", "d"],
            "Team": ["e", "e"],
            "Grant_ID": [1, 2],
            "Sciencetags": ["T1", "T2"],
        }
    )
    test_processed_data = [
        {"text": "A a", "meta": {"Grant_ID": 1, "Title": "A"}, "tags": ["T1"]},
        {"text": "B b", "meta": {"Grant_ID": 2, "Title": "B"}, "tags": ["T2"]},
    ]
    out = list(yield_preprocess_data(data))

    assert out == test_processed_data


def test_duplicate_grants():
    data = pd.DataFrame(
        {
            "Title": ["A", "A"],
            "Synopsis": ["a", "a"],
            "Lay Summary": ["b", "b"],
            "Qu.": ["c", "c"],
            "Scheme": ["d", "d"],
            "Team": ["e", "e"],
            "Grant_ID": [1, 1],
            "Sciencetags": ["T1", "T1"],
        }
    )
    test_processed_data = [
        {"text": "A a", "meta": {"Grant_ID": 1, "Title": "A"}, "tags": ["T1"]}
    ]
    out = list(yield_preprocess_data(data))

    assert out == test_processed_data


def test_missing_synopsis():
    data = pd.DataFrame(
        {
            "Title": ["A", "B"],
            "Synopsis": ["a", np.NaN],
            "Lay Summary": ["b", "b"],
            "Qu.": ["c", "c"],
            "Scheme": ["d", "d"],
            "Team": ["e", "e"],
            "Grant_ID": [1, 2],
            "Sciencetags": ["T1", "T2"],
        }
    )
    test_processed_data = [
        {"text": "A a", "meta": {"Grant_ID": 1, "Title": "A"}, "tags": ["T1"]}
    ]
    out = list(yield_preprocess_data(data))

    assert out == test_processed_data


def test_tagger1():
    data = pd.DataFrame(
        {
            "Title": ["A", "B"],
            "Synopsis": ["a", "b"],
            "Lay Summary": ["b", "b"],
            "Qu.": ["c", "c"],
            "Scheme": ["d", "d"],
            "Team": ["e", "e"],
            "Grant_ID": [1, 2],
            "Sciencetags": ["T1", "T2"],
            "intersection": ["[T1, T2]", "[]"],
            "Tagger 1 only": ["", "[T1]"],
        }
    )
    test_processed_data = [
        {"text": "A a", "meta": {"Tagger1_tags": ["T1", "T2"]}, "tags": ["T1"]},
        {"text": "B b", "meta": {"Tagger1_tags": ["T1"]}, "tags": ["T2"]},
    ]
    out = list(yield_preprocess_data(data, meta_cols=["Tagger1_tags"]))
    print(out)
    assert out == test_processed_data
