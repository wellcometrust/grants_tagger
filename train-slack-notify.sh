#!/usr/bin/env bash

# Do not send slack message if command fails
set -e

COMMAND=`echo $@`

curl -X POST -H 'Content-type: application/json' --data "{'text': 'Hi <$SLACK_USER>, your command \`${COMMAND}\` has started running'}" $SLACK_HOOK

$@

curl -X POST -H 'Content-type: application/json' --data "{'text': 'Hi <$SLACK_USER>, your command \`${COMMAND}\` has finished. :dancingbanana:'}" $SLACK_HOOK
