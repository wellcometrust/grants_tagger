#!/bin/bash

# For this script to work, you need to have the repository data-pipelines pulled,
# And place it in the root of that repository

python org.wellcome/mesh/files/opt/mesh/task.py \
        --input ../grants_tagger/data/raw/grants.csv \
        --output ../grants_tagger/data/interim/mesh_pipeline_result.csv \
	--threshold 0.01 \
	--approach mesh-xlinear \
        --model_path ../grants_tagger/models/xlinear/model \
        --label_binarizer_path ../grants_tagger/models/xlinear/label_binarizer.pkl
