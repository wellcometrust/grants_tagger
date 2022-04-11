import tempfile
import os.path
import csv

from grants_tagger.pretrain import pretrain


DATA = [{"synopsis": "One"}, {"synopsis": "Two"}, {"synopsis": "Three"}]


def test_pretrain_doc2vec():
    model_name = "doc2vec"

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = os.path.join(tmp_dir, "model")
        data_path = os.path.join(tmp_dir, "data.csv")

        with open(data_path, "w") as f:
            csvwriter = csv.DictWriter(f, fieldnames=["synopsis"])
            csvwriter.writeheader()
            for line in DATA:
                csvwriter.writerow(line)

        pretrain(data_path, model_path, model_name)
