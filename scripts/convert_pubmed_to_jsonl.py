import json
import argparse


def main(data_path: str, output_path: str):
    """
    Converts a JSON file containing PubMed abstracts and MeSH terms to a JSONL file with a specific format.
    The PubMed publications come from the Wellcome store.

    The input JSON file should be a list of dictionaries, where each dictionary represents a PubMed abstract and has the following keys:
    - "title": a string representing the title of the abstract
    - "abstract": a string representing the text of the abstract
    - "mesh_terms": a list of strings representing the MeSH terms associated with the abstract

    The output JSONL file will have one JSON object per line, where each object represents a PubMed abstract and has the following keys:
    - "text": a string representing the text of the abstract
    - "tags": a list of strings representing the MeSH terms associated with the abstract
    - "meta": a dictionary with metadata about the abstract, including the title

    Args:
        data_path (str): Path to the input JSON file.
        output_path (str): Path to the output JSONL file.

    Returns:
        None
    """
    with open(data_path) as f:
        data = json.load(f)

    data = [sample for sample in data if len(sample["mesh_terms"])]

    for pub in data:
        new_labels = []
        for label in pub["mesh_terms"]:
            if "/" in label:
                label = label.split("/", 1)[0]

            if label.endswith("*"):
                label = label[:-1]

            new_labels.append(label)
        pub["mesh_terms"] = new_labels

    for item in data:
        item["text"] = item.pop("abstract")
        item["tags"] = item.pop("mesh_terms")
        item["meta"] = {"title": item.pop("title")}

    with open(output_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--output-path", type=str)
    args = parser.parse_args()

    main(args.data_path, args.output_path)
