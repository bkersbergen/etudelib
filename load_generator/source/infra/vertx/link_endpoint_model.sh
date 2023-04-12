#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    echo "requires args 'VERTEX_ENDPOINT', 'VERTEX_MODEL'"
    exit 1
fi

export VERTEX_ENDPOINT="${1}"
export VERTEX_MODEL="${2}"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "vertex-link-endpoint-model-${VERTEX_ENDPOINT}-${VERTEX_MODEL}" --ignore-not-found=true --timeout=10m

envsubst < ./link_endpoint_model_job.yaml > "/tmp/vertex-link-endpoint-model-${VERTEX_ENDPOINT}-${VERTEX_MODEL}.yaml"
cat "/tmp/vertex-link-endpoint-model-${VERTEX_ENDPOINT}-${VERTEX_MODEL}.yaml"

#kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/vertex-link-endpoint-model-${VERTEX_ENDPOINT}-${VERTEX-MODEL}.yaml"
