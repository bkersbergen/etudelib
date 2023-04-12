#!/usr/bin/env bash
#gcloud ai models list --project=bolcom-pro-reco-analytics-fcc --region=europe-west4
OUTPUT=$(
gcloud ai endpoints list \
  --project=bolcom-pro-reco-analytics-fcc \
  --region=europe-west4 \
  --format='json'
&> /dev/null)

#echo "$OUTPUT" | jq .

echo "$OUTPUT" | jq '[.[] | {name:.name,display: .displayName, models: [.deployedModels[] | {name: .model, id: .id, display: .displayName, version: .modelVersionId, }]}]'

