#!/usr/bin/env bash

if [ $# -lt 3 ]; then
    echo "requires arg 'DEPLOY_IMAGE_URI MODEL_URI MODEL_NAME'"
    exit 1
fi



DEPLOY_IMAGE_URI="${1}"
MODEL_URI="${2}"
MODEL_NAME="${3}"
DIR="$(dirname "$0")"

function normalize() {
  echo "$1" | tr -cd '[:alnum:]' | tr '[:upper:]' '[:lower:]'
}


VERTEX_MODEL_IMAGE=$(normalize "${DEPLOY_IMAGE_URI}")
VERTEX_MODEL_NAME=$(normalize "${MODEL_NAME}")

echo "models['${VERTEX_MODEL_NAME}'].deploy(image = '${VERTEX_MODEL_IMAGE}')"

HASH=$(sum <<< "${VERTEX_MODEL_NAME}-${VERTEX_MODEL_IMAGE}" | cut -f 1 -d ' ')
JOB_NAME="vertex-deploy-model-${HASH}-$(date +%s)"

MODELS_STATE=$("$DIR"/gcloud/models_state.sh)

for MODEL in $(echo "$MODELS_STATE" | jq -r '.[].display'); do
    if [ "$MODEL" = "${VERTEX_MODEL_NAME}" ]; then
      echo "models['${VERTEX_MODEL_NAME}'].200"
      exit 0
    fi
done

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

export VERTEX_ENDPOINT_NAME MODEL_NAME JOB_NAME VERTEX_MODEL_IMAGE DEPLOY_IMAGE_URI MODEL_URI
envsubst < "$DIR"/deploy_model_job.yaml > "/tmp/deploy_model_job-${HASH}.yaml"
export -n VERTEX_ENDPOINT_NAME MODEL_NAME JOB_NAME VERTEX_MODEL_IMAGE DEPLOY_IMAGE_URI MODEL_URI

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/deploy_model_job-${HASH}.yaml"
POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
# Ignore any errors that may occur during the execution of the following kubectl (by appending `|| true`).
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=15m )

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow)
if [[ "$LOGS" =~ .*"Model created.".* ]]; then
  echo "models['${VERTEX_MODEL_NAME}'].deploy().ok"
  exit 0
fi

echo "$LOGS"
echo "models['${VERTEX_MODEL_NAME}'].deploy().err"
exit 1
