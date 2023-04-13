#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    echo "requires arg 'VERTEX_ENDPOINT_NAME, VERTEX_MODEL_NAME'"
    exit 1
fi


export VERTEX_ENDPOINT_NAME="${1}"
export VERTEX_MODEL_NAME="${2}"

echo "endpoints['${VERTEX_ENDPOINT_NAME}'].undeploy(model = '${VERTEX_MODEL_NAME}')"

HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME}-${VERTEX_MODEL_NAME}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-undeploy-endpoint-model-${HASH}-$(date +%s)"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=60s

envsubst < ./undeploy_endpoint_model_job.yaml > "/tmp/undeploy_endpoint_model_job.${VERTEX_ENDPOINT_NAME}_${VERTEX_MODEL_NAME}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/undeploy_endpoint_model_job.${VERTEX_ENDPOINT_NAME}_${VERTEX_MODEL_NAME}.yaml"
POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=5m)

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow)
[[ "$LOGS" =~ .*"Endpoint model undeployed.".* ]] && {
  echo "endpoints['${VERTEX_ENDPOINT_NAME}'].undeploy(model = '${VERTEX_MODEL_NAME}').ok"
  exit 0
}

echo "endpoints['${VERTEX_ENDPOINT_NAME}'].undeploy(model = '${VERTEX_MODEL_NAME}').err"
exit 1
