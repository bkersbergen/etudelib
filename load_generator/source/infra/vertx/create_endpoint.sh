#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_ENDPOINT_NAMNE'"
    exit 1
fi

export VERTEX_ENDPOINT_NAMNE="${1}"
HASH=$(sum <<< "${VERTEX_ENDPOINT_NAMNE}" | cut -f 1 -d ' ')
export JOB_NAME= "etude-vertex-create-endpoint-${HASH}"


kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=10m

envsubst < ./create_endpoint_job.yaml > "/tmp/create_endpoint_job.${VERTEX_ENDPOINT_NAMNE}.yaml"
# cat "/tmp/create_endpoint_job.${VERTEX_ENDPOINT_NAMNE}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/create_endpoint_job.${VERTEX_ENDPOINT_NAMNE}.yaml"
