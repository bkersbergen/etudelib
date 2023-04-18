#!/usr/bin/env bash
set -e

if [ $# -lt 3 ]; then
    echo "requires args 'VERTEX_ENDPOINT', 'CATALOG_SIZE', 'REPORT_URI'"
    exit 1
fi

VERTEX_ENDPOINT="${1}"
CATALOG_SIZE="${2}"
REPORT_URI="${3}"

HASH=$(sum <<< "${VERTEX_ENDPOINT}" | cut -f 1 -d ' ')
JOB_NAME="etude-run-loadtest-${HASH}-$(date +%s)"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

export JOB_NAME VERTEX_ENDPOINT CATALOG_SIZE REPORT_URI
envsubst < ./deploy_loadgen_job.yaml > "/tmp/deploy_loadgen_job.${HASH}.yaml"
export -n JOB_NAME VERTEX_ENDPOINT CATALOG_SIZE REPORT_URI

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/deploy_loadgen_job.${HASH}.yaml"


