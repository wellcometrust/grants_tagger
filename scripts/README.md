## ✍️ Scripts

This folder contains convenience functions to run certain tasks. It also contains
scripts to runs pipelines which are outside this repository, some of them in
private Wellcome repositories.

### Evaluation dag & getting grants
The main dvc DAG has an evaluation independent dag for one of the models (MeSH). To run this,
you need to be on the Wellcome VPN, you need to install private requirements
(`make install-private-requirements`).

### Trigger pipeline

`trigger_pipeline.sh` triggers our internal grant tagging job manually. For this to
run, at the moment there is a bit of manual config required. From the grants_tagger repository run:

```bash
cd ..
git clone git@github.com:wellcometrust/data-pipelines.git
cp grants_tagger/scripts/trigger_pipeline.sh data-pipelines
chmod +x data-pipelines/trigger_pipeline.sh
cd data-pipelines && ./trigger_pipeline.sh
```
❗ This will generate a file  `data/interim/mesh_pipeline_result.csv` by default. Change the path
if you don't want this to happen.
