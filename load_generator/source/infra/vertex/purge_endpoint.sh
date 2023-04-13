#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_ENDPOINT_NAME'"
    exit 1
fi

VERTEX_ENDPOINT_NAME="$1"
DIR="$(dirname "$0")"

HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME}" | cut -f 1 -d ' ')
JOB_NAME="vertex-delete-endpoint-${HASH}-$(date +%s)"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

VERTEX_ENDPOINT_NAME="${1}"
ENDPOINTS_STATE=$("$DIR"/gcloud/endpoints_state.sh)

for ENDPOINT in $(echo "$ENDPOINTS_STATE" | jq -r '.[].display'); do
    if [ "$ENDPOINT" = "$VERTEX_ENDPOINT_NAME" ]; then
      ENDPOINT_EXISTS=true
      echo "endpoint(name = '${VERTEX_ENDPOINT_NAME}').exist"
      break
    fi
done

[ "true" != "${ENDPOINT_EXISTS}" ] && {
   echo "endpoint(name = '${VERTEX_ENDPOINT_NAME}').404"
   exit 0
}

ENDPOINT_MODELS=$(echo "$ENDPOINTS_STATE" | jq "[.[] | select(.display == \"${VERTEX_ENDPOINT_NAME}\").models[]]")
echo "endpoint.models(length = $(echo "${ENDPOINT_MODELS}" | jq length))"

for model_id in $(echo "$ENDPOINT_MODELS" | jq -r '.[].id'); do
 echo "undeploy(model = ${model_id}, endpoint = ${VERTEX_ENDPOINT_NAME})"
 "$DIR"/undeploy_endpoint_model.sh "${VERTEX_ENDPOINT_NAME}" "${model_id}"
done

"$DIR"/delete_endpoint.sh "${VERTEX_ENDPOINT_NAME}"

[[ "$?" == "0" ]] && {
  echo "endpoint['${VERTEX_ENDPOINT_NAME}'].purge().ok"
  exit 0
}

echo "endpoint['${VERTEX_ENDPOINT_NAME}'].purge().err"
exit 1
