#!/usr/bin/env bash


OUTPUT=$(
gcloud ai endpoints list \
  --project=bolcom-pro-reco-analytics-fcc \
  --region=europe-west4 \
  --format='json' 2>/dev/null
)

echo "$OUTPUT" | jq '[.[] | {name:.name,display: .displayName, models: [.deployedModels[]? | {name: .model, id: .id, display: .displayName, version: .modelVersionId, }]}]'
