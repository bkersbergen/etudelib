#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_MODEL_NAME'"
    exit 1
fi

export VERTEX_MODEL_NAME="${1}"
HASH=$(sum <<< "${VERTEX_MODEL_NAME}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-deploy-model-${HASH}"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

envsubst < ./deploy_model_job.yaml > "/tmp/deploy_model_job.${VERTEX_MODEL_NAME}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/deploy_model_job.${VERTEX_MODEL_NAME}.yaml"
