#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    echo "requires args 'VERTEX_ENDPOINT_NAME', 'VERTEX_MODEL_NAME'"
    exit 1
fi

export VERTEX_ENDPOINT_NAME="${1}"
export VERTEX_MODEL_NAME="${2}"
HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME}-${VERTEX_MODEL_NAME}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-link-endpoint-model-${HASH}"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

envsubst < ./link_endpoint_model_job.yaml > "/tmp/vertex-link-endpoint-model-${VERTEX_ENDPOINT_NAME}-${VERTEX_MODEL_NAME}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/vertex-link-endpoint-model-${VERTEX_ENDPOINT_NAME}-${VERTEX-MODEL}.yaml"
