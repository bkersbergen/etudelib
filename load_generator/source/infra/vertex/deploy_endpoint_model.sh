#!/usr/bin/env bash

if [ $# -lt 2 ]; then
    echo "requires args 'VERTEX_ENDPOINT_NAME', 'VERTEX_MODEL_NAME'"
    exit 1
fi

export VERTEX_ENDPOINT_NAME="${1}"
export VERTEX_MODEL_NAME="${2}"

echo "endpoints['${VERTEX_ENDPOINT_NAME}'].deploy(model = '${VERTEX_MODEL_NAME}')"

HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME}-${VERTEX_MODEL_NAME}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-deploy-endpoint-model-${HASH}-$(date +%s)"

MODELS_STATE=$(./gcloud/models_state.sh)
MODEL_EXISTS=false

for MODEL in $(echo "$MODELS_STATE" | jq -r '.[].display'); do
    if [ "$MODEL" = "${VERTEX_MODEL_NAME}" ]; then
      MODEL_EXISTS=true
      break
    fi
done

[ "true" != "${MODEL_EXISTS}" ] && {
   echo "models['${VERTEX_MODEL_NAME}'].404"
   exit 0
}

ENDPOINTS_STATE=$(./gcloud/endpoints_state.sh)

MODEL_DEPLOYMENTS=$(echo "$ENDPOINTS_STATE" | jq -c "[.[] | select(.models[].display == \"${VERTEX_ENDPOINT_NAME}\")]")

[[ "0" != $(echo "${MODEL_DEPLOYMENTS}" | jq 'length') ]] && {
  echo "endpoints['${VERTEX_ENDPOINT_NAME}'].deployment(model = '${VERTEX_MODEL_NAME}').exists"
}

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

envsubst < ./deploy_endpoint_model_job.yaml > "/tmp/vertex-deploy-endpoint-model-${VERTEX_ENDPOINT_NAME}-${VERTEX_MODEL_NAME}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/vertex-deploy-endpoint-model-${VERTEX_ENDPOINT_NAME}-${VERTEX_MODEL_NAME}.yaml"

POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=5m)

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow)
[[ "$LOGS" =~ .*"Model created.".* ]] && {
  echo echo "models['${VERTEX_MODEL_NAME}'].deploy().ok"
  exit 0
}

echo "$LOGS"
echo "models['${VERTEX_MODEL_NAME}'].deploy().err"
exit 1
