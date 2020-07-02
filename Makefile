PROJECT_NAME := grants_tagger

PRIVATE_PROJECT_BUCKET := datalabs-data/$(PROJECT_NAME)
PUBLIC_PROJECT_BUCKET := datalabs-public/$(PROJECT_NAME)

VIRTUALENV := venv

.PHONY:sync_data
sync_data:
	aws s3 sync data/ s3://$(PRIVATE_PROJECT_BUCKET)/data/
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/data/raw data/raw --exclude "*allMeSH*"

.PHONY: sync_artifacts
sync_artifacts:
	echo "Sync processed data"
	aws s3 sync data/processed s3://$(PRIVATE_PROJECT_BUCKET)/data/processed 
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/data/processed data/processed
	echo "Sync models"
	aws s3 sync models/ s3://$(PRIVATE_PROJECT_BUCKET)/models/
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/models/ models/

.PHONY:virtualenv
virtualenv:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python python3 $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip install -r requirements.txt
	$(VIRTUALENV)/bin/pip install -r requirements_test.txt
	$(VIRTUALENV)/bin/pip install -e .

.PHONY: test
test:
	$(VIRTUALENV)/bin/pytest --disable-warnings -v --cov=grants_tagger

.PHONY: run_codecov
run_codecov:
	$(VIRTUALENV)/bin/python -m codecov

.PHONY: build
build:
	$(VIRTUALENV)/bin/python setup.py bdist_wheel

.PHONY: deploy
deploy:
	aws s3 cp --recursive --exclude "*" --include "*.whl" --acl public-read dist/ s3://$(PUBLIC_PROJECT_BUCKET)
