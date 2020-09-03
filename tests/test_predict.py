from grants_tagger.predict import predict_tags, predict_tags_fast

def test_predict_tags():
    tags = predict_tags("malaria")

def test_predict_tags_fast():
    tags = predict_tags_fast(["malaria", "ebola"])
