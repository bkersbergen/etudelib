#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_MODEL_NAME'"
    exit 1
fi

export VERTEX_MODEL_NAME="${1}"

echo "models['${VERTEX_MODEL_NAME}'].purge()"

HASH=$(sum <<< "${VERTEX_MODEL_NAME}" | cut -f 1 -d ' ')
export JOB_NAME="vertex-delete-model-${HASH}-$(date +%s)"

MODELS_STATE=$(./gcloud/models_state.sh)
MODEL_EXISTS=false

for MODEL in $(echo "$MODELS_STATE" | jq -r '.[].display'); do
    if [ "$MODEL" = "$VERTEX_MODEL_NAME" ]; then
      MODEL_EXISTS=true
      echo "models['${VERTEX_MODEL_NAME}'].200"
      break
    fi
done

[ "true" != "${MODEL_EXISTS}" ] && {
   echo "models['${VERTEX_MODEL_NAME}'].404"
   exit 0
}

ENDPOINTS_STATE=$(./gcloud/endpoints_state.sh)

MODEL_DEPLOYMENTS=$(echo "$ENDPOINTS_STATE" | jq -c "[.[] | select(.models[].display == \"${VERTEX_MODEL_NAME}\")]")

echo "models['${VERTEX_MODEL_NAME}'].deployments(length = $(echo "${ENDPOINT_MODELS}" | jq 'length'))"

for endpoint_model_deployment in $(echo "$MODEL_DEPLOYMENTS" | jq -c '.[] | {endpoint_name: .display, deployment_id: .models[0].id}'); do
  ENDPOINT_NAME=$(echo "${endpoint_model_deployment}" | jq -r '.endpoint_name')
  DEPLOYMENT_ID=$(echo "${endpoint_model_deployment}" | jq -r '.deployment_id')
   ./undeploy_endpoint_model.sh "${ENDPOINT_NAME}" "${DEPLOYMENT_ID}"
done

./delete_model.sh "${VERTEX_MODEL_NAME}"

[[ "$?" == "0" ]] && {
  echo "models['${VERTEX_MODEL_NAME}'].purge().ok"
  exit 0
}

echo "models['${VERTEX_MODEL_NAME}'].purge().err"
exit 1
