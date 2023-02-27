## v0.2.5
- fixed dependency bug
- decoupled xlinear and other models to relief the library requirements when installing the wheel
- updated setup.py file

## v0.2.4

- excludes terms that have the attribute "Do not use"
- excludes terms that we manually want to exclude using `descriptors_not_to_use_manual.csv`

## v0.2.2

- #180: Bug fix
## v0.2.1

Bug fixes:
- Fix download_model to download version specific mesh model
- #152 Fix grants_tagger package which was failing due to missing dependencies

## v0.2.0

Improvements:

- #76 Implements `grants_tagger download` which downloads data from EPMC that contain mesh tags
- #74 Implements `grants_tagger visualize` that uses streamlit to interact with the models
- #84 Implements `grants_tagger explain` that produces local and global explanations using SHAP
- #78 Extends `grants_tagger download` to download models from Github releases
- #90 Add `MeshXLinear` class, config and results for MeSH
- #100 Add time to train and ec2_instance information when training

## v0.1.x

This is the first release that packages everything developed
to this point. Grants tagger comes with a nice CLI, commands
are
* preprocess
* train
* evaluate
* predict
* tune
* pretrain
* tag
