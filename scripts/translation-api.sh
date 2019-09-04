#!/bin/bash

SRC=$1
TGT=$2
INPUT_FILE=$3
PROJECT_ID=$4
CRED_FILE=$5

if [ -z "$4" ];then
    export PROJECT_ID=$4
fi
if [ -z "$CRED_FILE" ];then
    export GOOGLE_APPLICATION_CREDENTIALS=$CRED_FILE
fi

data=()
while IFS= read -r line; do
    data+=(\'$line\')
done < $INPUT_FILE

contents="$(IFS=,; echo "${data[*]}")"

curl -X POST \
     -H "Authorization: Bearer "$(gcloud auth application-default print-access-token) \
     -H 'Content-Type: application/json' --data "{
     source_language_code: '$SRC',
     target_language_code: '$TGT',
     contents: [$contents]
 }" "https://translation.googleapis.com/v3beta1/projects/$PROJECT_ID/locations/global:translateText"

