#!/usr/bin/env bash

docker run -it --rm \
  -v ~/.config/gcloud:/home/pyxle/.config/gcloud:ro \
  eu.gcr.io/bolcom-stg-pyxle-images-043/vertexai-utilities \
  --project bolcom-pro-reco-analytics-fcc \
  "$@"
