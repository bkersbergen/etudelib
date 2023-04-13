#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_ENDPOINT_NAME_OR_ID'"
    exit 1
fi

HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME_OR_ID}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-delete-endpoint-${HASH}-$(date +%s)"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

export VERTEX_ENDPOINT_NAME_OR_ID="${1}"
ENDPOINTS_STATE=$(./gcloud/endpoints_state.sh)

for ENDPOINT in $(echo "$ENDPOINTS_STATE" | jq -r '.[].display'); do
    if [ "$ENDPOINT" = "$VERTEX_ENDPOINT_NAME_OR_ID" ]; then
      ENDPOINT_EXISTS=true
      echo "endpoint(name = '${VERTEX_ENDPOINT_NAME_OR_ID}').exist"
      break
    fi
done

[ "true" != "${ENDPOINT_EXISTS}" ] && {
   echo "endpoint(name = '${VERTEX_ENDPOINT_NAME_OR_ID}').404"
   exit 0
}

ENDPOINT_MODELS=$(echo "$ENDPOINTS_STATE" | jq "[.[] | select(.display == \"${VERTEX_ENDPOINT_NAME_OR_ID}\").models[]]")
echo "endpoint.models(length = $(echo "${ENDPOINT_MODELS}" | jq length))"

for model_id in $(echo "$ENDPOINT_MODELS" | jq -r '.[].id'); do
 echo "undeploy(model = ${model_id}, endpoint = ${VERTEX_ENDPOINT_NAME_OR_ID})"
 ./undeploy_endpoint_model.sh "${VERTEX_ENDPOINT_NAME_OR_ID}" "${model_id}"
done

./delete_endpoint.sh "${VERTEX_ENDPOINT_NAME_OR_ID}"

[[ "$?" == "0" ]] && {
  echo "endpoint['${VERTEX_ENDPOINT_NAME_OR_ID}'].purge().ok"
  exit 0
}

echo "endpoint['${VERTEX_ENDPOINT_NAME_OR_ID}'].purge().err"
exit 1

