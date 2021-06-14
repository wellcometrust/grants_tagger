#!/bin/bash

APPROACH=$1

JOB_NAME=$(aws sagemaker list-training-jobs \
	--name-contains $APPROACH | jq --raw-output '.TrainingJobSummaries[0].TrainingJobName')

mkdir /tmp/sagemaker_models/
aws s3 cp s3://$PROJECTS_BUCKET/$PROJECT_NAME/output/$JOB_NAME/output/model.tar.gz sagemaker_models/

cd /tmp/sagemaker_models/
tar -xzf model.tar.gz
rm model.tar.gz

aws s3 cp --recursive . s3://$PROJECTS_BUCKET/$PROJECT_NAME/models/

cd -
rm -rf /tmp/sagemaker_models/
