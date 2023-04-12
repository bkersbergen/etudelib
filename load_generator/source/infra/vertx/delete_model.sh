#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTX_MODEL_NAME_OR_ID'"
    exit 1
fi

export VERTX_MODEL_NAME_OR_ID="${1}"
HASH=$(sum <<< "${VERTX_MODEL_NAME_OR_ID}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-delete-model-${HASH}"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=10m

envsubst < ./delete_model_job.yaml > "/tmp/delete_model_job.${VERTX_MODEL_NAME_OR_ID}.yaml"
cat "/tmp/delete_model_job.${VERTX_MODEL_NAME_OR_ID}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/delete_model_job.${VERTX_MODEL_NAME_OR_ID}.yaml"