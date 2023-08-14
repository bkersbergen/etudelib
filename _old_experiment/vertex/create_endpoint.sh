#!/usr/bin/env bash


if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_ENDPOINT_NAME'"
    exit 1
fi

DIR="$(dirname "$0")"
VERTEX_ENDPOINT_NAME="${1}"
echo "endpoints['${VERTEX_ENDPOINT_NAME}'].create()"

HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME}" | cut -f 1 -d ' ')
JOB_NAME="vertex-create-endpoint-${HASH}-$(date +%s)"

ENDPOINTS_STATE=$("$DIR"/gcloud/endpoints_state.sh)

for ENDPOINT in $(echo "$ENDPOINTS_STATE" | jq -r '.[].display'); do
    if [ "$ENDPOINT" = "${VERTEX_ENDPOINT_NAME}" ]; then
      echo "endpoints['${VERTEX_ENDPOINT_NAME}'].200"
      exit 0
    fi
done

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

export VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME
envsubst < "$DIR"/create_endpoint_job.yaml > "/tmp/create_endpoint_job.${VERTEX_ENDPOINT_NAME}.yaml"
export -n VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/create_endpoint_job.${VERTEX_ENDPOINT_NAME}.yaml"
POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
# Ignore any errors that may occur during the execution of the following kubectl (by appending `|| true`).
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=5m )

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow )
if [[ "$LOGS" =~ .*"Endpoint created.".* ]]; then
  echo "endpoints['${VERTEX_ENDPOINT_NAME}'].create().ok"
  exit 0
fi

echo "$LOGS"
echo "endpoints['${VERTEX_ENDPOINT_NAME}'].create().err"
exit 1
