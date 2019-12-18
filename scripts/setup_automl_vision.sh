#!/bin/bash

CONFIGURATION_NAME=$1
SERVICE_ACCOUNT_MAIL=$2
KEY_PATH=$3

gcloud config configurations activate $CONFIGURATION_NAME
if [ $? -ne 0 ]; then
    echo "$CONFIGURATION_NAME does not exist"
    exit 1
fi

if [ ! -e $KEY_PATH ]; then
    echo "$KEY_PATH does not exist"
    exit 1
fi

export GOOGLE_APPLICATION_CREDENTIALS=KEY_PATH

PROJECT=$(gcloud config get-value project)

gcloud projects add-iam-policy-binding $PROJECT \
       --member=serviceAccount:$SERVICE_ACCOUNT_MAIL \
       --role='roles/automl.editor'

gcloud projects add-iam-policy-binding $PROJECT \
   --member="serviceAccount:custom-vision@appspot.gserviceaccount.com" \
   --role="roles/ml.admin"
gcloud projects add-iam-policy-binding $PROJECT \
   --member="serviceAccount:custom-vision@appspot.gserviceaccount.com" \
   --role="roles/storage.admin"

BUCKET="${PROJECT}-vcm"
gsutil mb -p ${PROJECT} -c regional -l us-central1 gs://${BUCKET}

