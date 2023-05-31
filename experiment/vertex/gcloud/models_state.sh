#!/usr/bin/env bash


OUTPUT=$(
gcloud ai models list \
  --project=bolcom-pro-reco-analytics-fcc \
  --region=europe-west4 \
  --format='json' 2>/dev/null
)

echo "$OUTPUT" | jq '[.[] | {name:.name,display: .displayName, version: .versionId, image: .containerSpec.imageUri}]'
