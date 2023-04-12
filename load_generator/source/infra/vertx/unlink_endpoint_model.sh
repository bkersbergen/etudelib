#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    echo "requires arg 'VERTX_ENDPOINT_NAME_OR_ID, VERTX_MODEL_ID'"
    exit 1
fi

export VERTX_ENDPOINT_NAME_OR_ID="${1}"
export VERTX_MODEL_ID="${2}"
HASH=$(sum <<< "${VERTX_ENDPOINT_NAME_OR_ID}-${VERTX_MODEL_ID}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-unlink-endpoint-model-${HASH}"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=10m

envsubst < ./unlink_endpoint_model_job.yaml > "/tmp/unlink_endpoint_model_job.${VERTX_ENDPOINT_NAME_OR_ID}_${VERTX_MODEL_ID}.yaml"
cat "/tmp/unlink_endpoint_model_job.${VERTX_ENDPOINT_NAME_OR_ID}_${VERTX_MODEL_ID}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/unlink_endpoint_model_job.${VERTX_ENDPOINT_NAME_OR_ID}_${VERTX_MODEL_ID}.yaml"
