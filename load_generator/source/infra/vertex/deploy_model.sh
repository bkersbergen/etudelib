#!/usr/bin/env bash

if [ $# -lt 1 ]; then
    echo "requires arg 'VERTEX_MODEL_NAME'"
    exit 1
fi

export VERTEX_MODEL_NAME="${1}"
echo "models['${VERTEX_MODEL_NAME}'].deploy()"

HASH=$(sum <<< "${VERTEX_MODEL_NAME}" | cut -f 1 -d ' ')
export JOB_NAME="etude-vertex-deploy-model-${HASH}-$(date +%s)"

MODELS_STATE=$(./gcloud/models_state.sh)

for MODEL in $(echo "$MODELS_STATE" | jq -r '.[].display'); do
    if [ "$MODEL" = "$VERTEX_MODEL_NAME_OR_ID" ]; then
      echo "models['${VERTEX_MODEL_NAME_OR_ID}'].exists"
      exit 0
    fi
done

kubectl --context bolcom-pro-default --namespace reco-analytics delete job "${JOB_NAME}" --ignore-not-found=true --timeout=5m

envsubst < ./deploy_model_job.yaml > "/tmp/deploy_model_job.${VERTEX_MODEL_NAME}.yaml"

kubectl --context bolcom-pro-default --namespace reco-analytics apply --namespace reco-analytics -f - < "/tmp/deploy_model_job.${VERTEX_MODEL_NAME}.yaml"

POD_NAME=$(kubectl get pods --context bolcom-pro-default --namespace reco-analytics -l job-name="$JOB_NAME" -o custom-columns=:metadata.name | tr -d '\n')
POD_READY=$(kubectl --context bolcom-pro-default --namespace reco-analytics wait --for=condition=Ready pod/"$POD_NAME" --timeout=5m)

LOGS=$(kubectl --context bolcom-pro-default --namespace reco-analytics logs pod/"${POD_NAME}" --follow)
[[ "$LOGS" =~ .*"Model deployed.".* ]] && {
  echo echo "models['${VERTEX_MODEL_NAME}'].deploy().ok"
  exit 0
}

echo "$LOGS"
echo "models['${VERTEX_MODEL_NAME}'].deploy().err"
exit 1
