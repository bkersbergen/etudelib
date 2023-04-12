#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'OBJECT_TYPE'"
    exit 1
fi

export OBJECT_TYPE="${1}"
HASH=$(sum <<< "${OBJECT_TYPE}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-list-object-${HASH}"


kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=10m

envsubst < ./list_objects_job.yaml > "/tmp/list_objects_job.yaml"
# cat "/tmp/list_objects_job.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/list_objects_job.yaml"
clear

kubectl  --context bolcom-pro-default --namespace reco-analytics  wait --for=condition=complete job/"${JOB_NAME}" --timeout=60s
OUTPUT=$(kubectl --context bolcom-pro-default --namespace reco-analytics  logs job/"${JOB_NAME}")

echo "${OUTPUT}" | grep "^\\|.*" | tail -n +4 | head -n -1 | awk -F'\│' '{print $2, $3, $5}' | trim
