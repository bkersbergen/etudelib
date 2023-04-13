#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'OBJECT_TYPE'"
    exit 1
fi

export OBJECT_TYPE="${1}"

docker run -it --rm \
  -v ~/.config/gcloud:/home/pyxle/.config/gcloud:ro \
  eu.gcr.io/bolcom-stg-pyxle-images-043/vertexai-utilities \
  --project bolcom-pro-reco-analytics-fcc \
  "$@" list

echo "$(docker run -it --rm \
  -v ~/.config/gcloud:/home/pyxle/.config/gcloud:ro \
  eu.gcr.io/bolcom-stg-pyxle-images-043/vertexai-utilities \
  --project bolcom-pro-reco-analytics-fcc \
  "$@" list)"
#
#OUTPUT=$(../vertex.sh "${OBJECT_TYPE}" list)
#
#echo "${OUTPUT}"
#echo  "$OUTPUT" | grep "^\\|.*" | tail -n +4 | head -n -1 | awk -F'\│' '{print $2, $3, $5}' | grep "\S"
