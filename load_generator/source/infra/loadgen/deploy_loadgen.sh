#!/usr/bin/env bash

export JOB_NAME=${1:-???}

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=10m

envsubst < ./deploy_loadgen_job.yaml > "/tmp/deploy_loadgen_job.${JOB_NAME}.yaml"
# cat "/tmp/deploy_loadgen_job.${JOB_NAME}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/deploy_loadgen_job.${JOB_NAME}.yaml"


