#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'OBJECT_TYPE'"
    exit 1
fi

export OBJECT_TYPE="${1}"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "etude-list-objects" --ignore-not-found=true --timeout=10m

envsubst < ./list_objects_job.yaml > "/tmp/list_objects_job.yaml"

#cat "/tmp/list_objects_job.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/list_objects_job.yaml"
