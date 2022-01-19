import typer

from grants_tagger.predict import predict_tags


def tag_bioasq_data(data_path, model_path, label_binarizer_path, tagged_data_path, threshold=0.5):
    data = json.load(data_path)
    X = [item["Title"] + " " + item["abstractText"] for item in data]
    pmids = [item["pmid"] for item in data]

    tags = predict_tags(X, model_path, label_binarizer_path, "mesh-xlinear", threshold=threshold)
    
    tagged_data = [{"pmid": pmid, "labels": tags_} for pmid, tags_ in zip(pmids, tags)]
   
    json.dump(tagged_data, tagged_data_path) 

if __name__ == "__main__":
    typer.run(tag_bioasq_data)
