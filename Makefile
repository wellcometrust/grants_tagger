PROJECT_NAME := grants_tagger

PRIVATE_PROJECT_BUCKET := datalabs-data/$(PROJECT_NAME)
PUBLIC_PROJECT_BUCKET := datalabs-public/$(PROJECT_NAME)

VIRTUALENV := venv

.PHONY:sync_data
sync_data:
	aws s3 sync data/raw/ s3://$(PRIVATE_PROJECT_BUCKET)/data/raw/
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/data/raw data/raw --exclude "*allMeSH*"

.PHONY: sync_artifacts
sync_artifacts:
	echo "Sync processed data"
	aws s3 sync data/processed s3://$(PRIVATE_PROJECT_BUCKET)/data/processed 
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/data/processed data/processed
	echo "Sync models"
	aws s3 sync models/ s3://$(PRIVATE_PROJECT_BUCKET)/models/
	aws s3 sync s3://$(PRIVATE_PROJECT_BUCKET)/models/ models/

$(VIRTUALENV)/.installed:
	@if [ -d $(VIRTUALENV) ]; then rm -rf $(VIRTUALENV); fi
	@mkdir -p $(VIRTUALENV)
	virtualenv --python python3.7 $(VIRTUALENV)
	$(VIRTUALENV)/bin/pip install -r requirements.txt
	$(VIRTUALENV)/bin/pip install -r requirements_test.txt
	$(VIRTUALENV)/bin/pip install -e .
	$(VIRTUALENV)/bin/pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
	touch $@

.PHONY: virtualenv
virtualenv: $(VIRTUALENV)/.installed


.PHONY: test
test: virtualenv
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

.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*flymake*" -delete
	find . -type f -name "#*#" -delete
