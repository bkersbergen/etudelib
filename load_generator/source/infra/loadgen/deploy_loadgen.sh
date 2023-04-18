#!/usr/bin/env bash
set -e

if [ $# -lt 3 ]; then
    echo "requires args 'LOADTEST_NAME', 'RUNTIME', 'CATALOG_SIZE'"
    exit 1
fi

LOADTEST_NAME="${1}"
RUNTIME="${2}"
CATALOG_SIZE="${2}"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "etude-${LOADTEST_NAME}" --ignore-not-found=true --timeout=5m

export LOADTEST_NAME RUNTIME CATALOG_SIZE
envsubst < ./deploy_loadgen_job.yaml > "/tmp/deploy_loadgen_job.${LOADTEST_NAME}.yaml"
export -n LOADTEST_NAME RUNTIME CATALOG_SIZE

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/deploy_loadgen_job.${LOADTEST_NAME}.yaml"


