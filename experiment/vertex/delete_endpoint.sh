#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_ENDPOINT_NAME'"
    exit 1
fi

VERTEX_ENDPOINT_NAME="${1}"
DIR="$(dirname "$0")"

echo "endpoints['${VERTEX_ENDPOINT_NAME}'].delete()"

ENDPOINTS_STATE=$("$DIR"/gcloud/endpoints_state.sh)

for ENDPOINT in $(echo "$ENDPOINTS_STATE" | jq -r '.[].display'); do
    if [ "$ENDPOINT" = "$VERTEX_ENDPOINT_NAME" ]; then
      ENDPOINT_EXISTS=true
      break
    fi
done

if [ "true" != "${ENDPOINT_EXISTS}" ]; then
   echo "endpoints['${VERTEX_ENDPOINT_NAME}'].404"
   exit 0
fi

HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME}" | cut -f 1 -d ' ')
JOB_NAME="vertex-delete-endpoint-${HASH}-$(date +%s)"

export VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME
envsubst < "$DIR"/delete_endpoint_job.yaml > "/tmp/delete_endpoint_job.${VERTEX_ENDPOINT_NAME}.yaml"
export -n VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/delete_endpoint_job.${VERTEX_ENDPOINT_NAME}.yaml"
POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
# Ignore any errors that may occur during the execution of the following kubectl (by appending `|| true`).
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=5m 2> error.log || true)

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow 2> error.log || true)
if [[ "$LOGS" =~ .*"Endpoint deleted.".* ]]; then
  echo "endpoints['${VERTEX_ENDPOINT_NAME}'].delete().ok"
  exit 0
fi


echo "${LOGS}"
echo "endpoints['${VERTEX_ENDPOINT_NAME}'].delete().err"
exit 1
