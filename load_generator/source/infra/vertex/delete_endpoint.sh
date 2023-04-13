#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_ENDPOINT_NAME_OR_ID'"
    exit 1
fi

export VERTEX_ENDPOINT_NAME_OR_ID="${1}"

echo "endpoints['${VERTEX_ENDPOINT_NAME_OR_ID}'].delete()"

ENDPOINTS_STATE=$(./gcloud/endpoints_state.sh)

for ENDPOINT in $(echo "$ENDPOINTS_STATE" | jq -r '.[].display'); do
    if [ "$ENDPOINT" = "$VERTEX_ENDPOINT_NAME_OR_ID" ]; then
      ENDPOINT_EXISTS=true
      break
    fi
done

[ "true" != "${ENDPOINT_EXISTS}" ] && {
   echo "endpoints['${VERTEX_ENDPOINT_NAME_OR_ID}'].404"
   exit 0
}

HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME_OR_ID}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-delete-endpoint-${HASH}-$(date +%s)"

envsubst < ./delete_endpoint_job.yaml > "/tmp/delete_endpoint_job.${VERTEX_ENDPOINT_NAME_OR_ID}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/delete_endpoint_job.${VERTEX_ENDPOINT_NAME_OR_ID}.yaml"

POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=5m)

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow)
[[ "$LOGS" =~ .*"Endpoint deleted.".* ]] && {
  echo "endpoints['${VERTEX_ENDPOINT_NAME_OR_ID}'].delete().ok"
  exit 0
}

echo "endpoints['${VERTEX_ENDPOINT_NAME_OR_ID}'].delete().err"
exit 1
