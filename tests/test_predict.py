from grants_tagger.predict import predict_tags


def test_predict_tags():
    tags = predict_tags("This is a grant about malaria", "Wellcome/WellcomeBertMesh")
    tags = tags[0]

    assert "Malaria" in tags
    assert "Neoplasms" not in tags
    assert len(tags) < 10
