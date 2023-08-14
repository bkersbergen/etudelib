#!/usr/bin/env bash


if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_MODEL_NAME'"
    exit 1
fi

VERTEX_MODEL_NAME="${1}"
DIR="$(dirname "$0")"

echo "models['${VERTEX_MODEL_NAME}'].delete()"

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

HASH=$(sum <<< "${VERTEX_MODEL_NAME}" | cut -f 1 -d ' ')
JOB_NAME="vertex-delete-model-${HASH}-$(date +%s)"

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

export VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME
envsubst < "$DIR"/delete_model_job.yaml > "/tmp/delete_model_job.${HASH}.yaml"
export -n VERTEX_ENDPOINT_NAME VERTEX_MODEL_NAME JOB_NAME

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/delete_model_job.${HASH}.yaml"
POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
# Ignore any errors that may occur during the execution of the following kubectl (by appending `|| true`).
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=5m )

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow )
if [[ "$LOGS" =~ .*"Model deleted.".* ]]; then
  echo "models['${VERTEX_MODEL_NAME}'].delete().ok"
  exit 0
fi

echo "$LOGS"
echo "models['${VERTEX_MODEL_NAME}'].delete().err"
exit 1
