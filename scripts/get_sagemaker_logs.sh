#!/bin/bash

APPROACH=$1

LOG_STREAM_NAME=$(aws logs describe-log-streams \
	--log-group-name /aws/sagemaker/TrainingJobs \
	--descending --log-stream-name-prefix $APPROACH | jq --raw-output '.logStreams[0].logStreamName')
echo $LOG_STREAM_NAME

aws logs get-log-events \
	--log-group-name /aws/sagemaker/TrainingJobs \
	--log-stream-name $LOG_STREAM_NAME | jq '.events[].message'
