import xml.etree.ElementTree as ET
import typer
from tqdm import tqdm
import json

from grants_tagger.predict import predict_tags


def get_mesh2code(mesh_metadata_path):
    mesh_tree = ET.parse(mesh_metadata_path)

    mesh2code = {}
    for mesh in tqdm(mesh_tree.getroot()):
        try:
            # DescriptorUI e.g. M000616943
            mesh_code = mesh[0].text
            # DescriptorName e.g. Mucosal-Associated Invariant T Cells
            mesh_name = mesh[1][0].text
            
            mesh2code[mesh_name] = mesh_code
        except IndexError:
            pass
    return mesh2code

def tag_bioasq_data(data_path, model_path, label_binarizer_path, mesh_metadata_path, tagged_data_path, threshold=0.5):
    with open(data_path) as f:
        data = json.load(f)

    mesh2code = get_mesh2code(mesh_metadata_path)

    X = []
    pmids = []
    for item in tqdm(data):
        X.append(item["title"] + " " + item["abstractText"])
        pmids.append(item["pmid"])

    tags = predict_tags(X, model_path, label_binarizer_path, "mesh-xlinear", threshold=threshold)
    
    tagged_data = [{"pmid": pmid, "labels": [mesh2code[t] for t in tags_]} for pmid, tags_ in zip(pmids, tags)]
  
    with open(tagged_data_path, "w") as f:
        json.dump({"documents": tagged_data}, f) 

if __name__ == "__main__":
    typer.run(tag_bioasq_data)
