from science_tagger.preprocess_mesh import process_data

def test_process_data():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"]
    }
    expected_processed_item = {
        "text": "This is an abstract",
        "tags": ["T1", "T2"],
        "meta": {}
    }
    processed_item = process_data(item)
    assert processed_item == expected_processed_item

def test_process_data_with_filter_tags():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"]
    }
    expected_processed_item = {
        "text": "This is an abstract",
        "tags": ["T1"],
        "meta": {}
    }
    processed_item = process_data(item, filter_tags=["T1"])
    assert processed_item == expected_processed_item

def test_process_data_with_missing_filter_tag():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"]
    }
    processed_item = process_data(item, filter_tags=["T3"])
    assert processed_item == None
