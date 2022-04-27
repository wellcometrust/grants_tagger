from grants_tagger.bertmesh.model import BertMesh

# this script requires you login with the cli first with huggingface-cli login

# this requires transformers >= 4.17 and allows us to push model.py as well
BertMesh.register_for_auto_class("AutoModel")

model = BertMesh.from_pretrained("models/bertmesh/model")
model.push_to_hub(
    "WellcomeBertMesh",
    organization="Wellcome",
    repo_url="https://huggingface.co/Wellcome/WellcomeBertMesh",
    #    use_temp_dir=True # for some reason this yield an authentication error
)

# note we do not push the tokenizer as we use bert pretrained one so it is unlikely to change
