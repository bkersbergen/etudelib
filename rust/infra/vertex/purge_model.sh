#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_MODEL_NAME'"
    exit 1
fi

VERTEX_MODEL_NAME="${1}"
DIR="$(dirname "$0")"

echo "models['${VERTEX_MODEL_NAME}'].purge()"

HASH=$(sum <<< "${VERTEX_MODEL_NAME}" | cut -f 1 -d ' ')
JOB_NAME="vertex-delete-model-${HASH}-$(date +%s)"

MODELS_STATE=$("$DIR"/gcloud/models_state.sh)
MODEL_EXISTS=false

for MODEL in $(echo "$MODELS_STATE" | jq -r '.[].display'); do
    if [ "$MODEL" = "$VERTEX_MODEL_NAME" ]; then
      MODEL_EXISTS=true
      break
    fi
done

if [ "true" != "${MODEL_EXISTS}" ]; then
   echo "models['${VERTEX_MODEL_NAME}'].404"
   exit 0
fi

ENDPOINTS_STATE=$("$DIR"/gcloud/endpoints_state.sh)

MODEL_DEPLOYMENTS=$(echo "$ENDPOINTS_STATE" | jq -c "[.[] | select(.models[].display == \"${VERTEX_MODEL_NAME}\")]")

echo "models['${VERTEX_MODEL_NAME}'].deployments(length = $(echo "${MODEL_DEPLOYMENTS}" | jq 'length'))"

for DEPLOYMENT in $(echo "$MODEL_DEPLOYMENTS" | jq -c '.[] | .display'); do
   VERTEX_ENDPOINT_NAME=$(echo "${DEPLOYMENT}" | jq -r .)
  "$DIR"/undeploy_endpoint_model.sh "${VERTEX_ENDPOINT_NAME}" "${VERTEX_MODEL_NAME}"
done

"$DIR"/delete_model.sh "${VERTEX_MODEL_NAME}"

if [[ "$?" == "0" ]]; then
  echo "models['${VERTEX_MODEL_NAME}'].purge().ok"
  exit 0
fi

echo "models['${VERTEX_MODEL_NAME}'].purge().err"
exit 1
