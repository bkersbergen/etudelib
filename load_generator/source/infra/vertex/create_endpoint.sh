#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_ENDPOINT_NAME'"
    exit 1
fi

export VERTEX_ENDPOINT_NAME="${1}"
echo "endpoints['${VERTEX_ENDPOINT_NAME}'].create()"


HASH=$(sum <<< "${VERTEX_ENDPOINT_NAME}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-create-endpoint-${HASH}-$(date +%s)"

ENDPOINTS_STATE=$(./gcloud/endpoints_state.sh)

for ENDPOINT in $(echo "$ENDPOINTS_STATE" | jq -r '.[].display'); do
    if [ "$ENDPOINT" = "${VERTEX_ENDPOINT_NAME}" ]; then
      echo "endpoints['${VERTEX_ENDPOINT_NAME}'].exist"
      exit 0
    fi
done

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

envsubst < ./create_endpoint_job.yaml > "/tmp/create_endpoint_job.${VERTEX_ENDPOINT_NAME}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/create_endpoint_job.${VERTEX_ENDPOINT_NAME}.yaml"

POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=5m)

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow)

[[ "$LOGS" =~ .*"Endpoint created.".* ]] && {
  echo "endpoints['${VERTEX_ENDPOINT_NAME}'].create().ok"
  exit 0
}

echo "$LOGS"
echo "endpoints['${VERTEX_ENDPOINT_NAME}'].create().err"
exit 1
