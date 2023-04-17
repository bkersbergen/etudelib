#!/usr/bin/env bash
set -e

if [ $# -lt 2 ]; then
    echo "requires arg 'VERTEX_MODEL_NAME'"
    exit 1
fi

VERTEX_MODEL_NAME="${1}"
VERTEX_MODEL_IMAGE="${2}"
DIR="$(dirname "$0")"

echo "models['${VERTEX_MODEL_NAME}'].deploy()"

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

export VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME VERTEX_MODEL_IMAGE
envsubst < "$DIR"/deploy_model_job.yaml > "/tmp/deploy_model_job.${VERTEX_MODEL_NAME}-${HASH}.yaml"
export -n VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME VERTEX_MODEL_IMAGE

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/deploy_model_job.${VERTEX_MODEL_NAME}-${HASH}.yaml"
POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=15m)

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow)
[[ "$LOGS" =~ .*"Model created.".* ]] && {
  echo "models['${VERTEX_MODEL_NAME}'].deploy().ok"
  exit 0
}

echo "$LOGS"
echo "models['${VERTEX_MODEL_NAME}'].deploy().err"
exit 1
