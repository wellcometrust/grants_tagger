import json

import typer


def yield_labels(label_data_path):
    with open(label_data_path, encoding='latin-1') as f:
        for line in f:
            yield line


def yield_processed_data(raw_data_path, labels):
    with open(raw_data_path) as f:
        for line in f:
            item = json.loads(line)
            text = item["title"] + " " + item["content"]
            tags = [labels[label_index] for label_index in item["target_ind"]]
            yield {"text": text, "tags": tags}


def preprocess_amazon(raw_data_path, label_data_path, processed_data_path):
    labels = list(yield_labels(label_data_path))

    with open(processed_data_path, "w") as f:
        for item in yield_processed_data(raw_data_path, labels):
            f.write(json.dumps(item))
            f.write("\n")


if __name__ == "__main__":
    typer.run(preprocess_amazon)
