#!/usr/bin/env bash
#gcloud ai models list --project=bolcom-pro-reco-analytics-fcc --region=europe-west4
OUTPUT=$(
gcloud ai models list \
  --project=bolcom-pro-reco-analytics-fcc \
  --region=europe-west4 \
  --format='json'
&> /dev/null)

#echo "$OUTPUT" | jq .

echo "$OUTPUT" | jq '[.[] | {name:.name,display: .displayName, version: .versionId, image: .containerSpec.imageUri}]'

