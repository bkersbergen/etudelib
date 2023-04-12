#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_ENDPOINT'"
    exit 1
fi

export VERTEX_ENDPOINT="${1}"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "vertex-create-endpoint-${VERTEX_ENDPOINT}" --ignore-not-found=true --timeout=10m

envsubst < ./create_endpoint_job.yaml > "/tmp/create_endpoint_job.${VERTEX_ENDPOINT}.yaml"
# cat "/tmp/create_endpoint_job.${VERTEX_ENDPOINT}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/create_endpoint_job.${VERTEX_ENDPOINT}.yaml"
