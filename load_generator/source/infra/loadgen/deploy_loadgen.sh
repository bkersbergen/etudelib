#!/usr/bin/env bash


if [ $# -lt 1 ]; then
    echo "requires args 'LOADTEST_NAME'"
    exit 1
fi

export LOADTEST_NAME="${1}"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "etude-${LOADTEST_NAME}" --ignore-not-found=true --timeout=5m

envsubst < ./deploy_loadgen_job.yaml > "/tmp/deploy_loadgen_job.${LOADTEST_NAME}.yaml"
 cat "/tmp/deploy_loadgen_job.${LOADTEST_NAME}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/deploy_loadgen_job.${LOADTEST_NAME}.yaml"


