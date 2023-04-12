#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_MODEL'"
    exit 1
fi

export VERTX_MODEL="${1}"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "etude-vertex-deploy-model-${VERTX_MODEL}" --ignore-not-found=true --timeout=10m

envsubst < ./deploy_model_job.yaml > "/tmp/deploy_model_job.${VERTX_MODEL}.yaml"
# cat "/tmp/deploy_model_job.${VERTX_MODEL}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/deploy_model_job.${VERTX_MODEL}.yaml"
