#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_ENDPOINT_NAME_OR_ID'"
    exit 1
fi

HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME_OR_ID}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-delete-endpoint-${HASH}-$(date +%s)"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=10m

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
   exit 1
}

ENDPOINT_MODELS=$(echo "$ENDPOINTS_STATE" | jq "[.[] | select(.display == \"${VERTEX_ENDPOINT_NAME_OR_ID}\").models[]]")
# echo "$ENDPOINT_MODELS" | jq .
echo "endpoint.models(length = $(echo "${ENDPOINT_MODELS}" | jq length))"

for model_id in $(echo "$ENDPOINT_MODELS" | jq -r '.[].id'); do
 echo "unlink(model = ${model_id}, endpoint = ${VERTEX_ENDPOINT_NAME_OR_ID})"
 ./unlink_endpoint_model.sh "${VERTEX_ENDPOINT_NAME_OR_ID}" "${model_id}"
done

envsubst < ./delete_endpoint_job.yaml > "/tmp/delete_endpoint_job.${VERTEX_ENDPOINT_NAME_OR_ID}.yaml"
# cat "/tmp/delete_endpoint_job.${VERTEX_ENDPOINT_NAME_OR_ID}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/delete_endpoint_job.${VERTEX_ENDPOINT_NAME_OR_ID}.yaml"

echo "delete(endpoint = ${VERTEX_ENDPOINT_NAME_OR_ID})"
POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=60s)

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow)
[[ "$LOGS" =~ .*"Endpoint deleted.".* ]] && {
  echo "ok"
  exit 0
}

echo "nok"
exit 1
